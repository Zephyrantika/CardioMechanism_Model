"""Leiden module detection, perturbation matching, and enrichment."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import igraph as ig
import leidenalg
import numpy as np
import pandas as pd
from scipy.stats import hypergeom
from statsmodels.stats.multitest import multipletests


def detect_leiden_modules(
    gene_ids: Sequence[str],
    ppi_edges: pd.DataFrame,
    *,
    seed: int,
    minimum_size: int = 5,
    maximum_size: int = 100,
) -> list[tuple[str, ...]]:
    """Detect deterministic weighted Leiden modules in an induced gene subgraph."""
    if minimum_size < 2 or maximum_size < minimum_size:
        raise ValueError("Invalid module size limits")
    genes = tuple(dict.fromkeys(map(str, gene_ids)))
    index = {gene_id: position for position, gene_id in enumerate(genes)}
    selected = ppi_edges.loc[
        ppi_edges["gene_a"].astype(str).isin(index)
        & ppi_edges["gene_b"].astype(str).isin(index)
    ].copy()
    graph = ig.Graph(n=len(genes), directed=False)
    graph.vs["name"] = list(genes)
    if not selected.empty:
        graph.add_edges([
            (index[str(row.gene_a)], index[str(row.gene_b)])
            for row in selected.itertuples(index=False)
        ])
        graph.es["weight"] = selected["weight"].astype(float).tolist()
    if graph.ecount() == 0:
        return []
    partition = leidenalg.find_partition(
        graph,
        leidenalg.RBConfigurationVertexPartition,
        weights="weight",
        seed=int(seed),
    )
    modules = []
    for membership in partition:
        members = tuple(sorted((genes[position] for position in membership), key=int))
        if minimum_size <= len(members) <= maximum_size:
            modules.append(members)
    return sorted(modules, key=lambda values: (-len(values), tuple(map(int, values))))


def module_topology(
    members: Sequence[str],
    ppi_edges: pd.DataFrame,
) -> dict[str, float | int | bool]:
    """Summarize weighted internal connectivity and hub dominance."""
    member_set = set(map(str, members))
    selected = ppi_edges.loc[
        ppi_edges["gene_a"].astype(str).isin(member_set)
        & ppi_edges["gene_b"].astype(str).isin(member_set)
    ]
    degree = {gene_id: 0.0 for gene_id in member_set}
    for row in selected.itertuples(index=False):
        weight = float(row.weight)
        degree[str(row.gene_a)] += weight
        degree[str(row.gene_b)] += weight
    total_degree = sum(degree.values())
    hub_dominance = (
        2.0 * max(degree.values(), default=0.0) / total_degree
        if total_degree
        else 1.0
    )
    possible = len(member_set) * (len(member_set) - 1) / 2
    return {
        "internal_edge_count": len(selected),
        "internal_density": len(selected) / possible if possible else 0.0,
        "mean_internal_weight": float(selected["weight"].mean()) if len(selected) else 0.0,
        "hub_dominance": hub_dominance,
        "single_hub_dominated": bool(hub_dominance > 0.5),
    }


def match_module_stability(
    base_members: Sequence[str],
    perturbed_modules: Sequence[Sequence[Sequence[str]]],
) -> tuple[float, dict[str, float]]:
    """Match a base module by maximum Jaccard in each run and score members."""
    base = set(map(str, base_members))
    if not perturbed_modules:
        raise ValueError("At least one perturbation run is required")
    jaccards = []
    occurrences = {gene_id: 0 for gene_id in base}
    for run_modules in perturbed_modules:
        candidates = [set(map(str, module)) for module in run_modules]
        matched = max(
            candidates,
            key=lambda module: (
                len(base & module) / len(base | module) if base | module else 0.0,
                len(base & module),
                -len(module),
            ),
            default=set(),
        )
        jaccards.append(len(base & matched) / len(base | matched) if base | matched else 0.0)
        for gene_id in base & matched:
            occurrences[gene_id] += 1
    runs = len(perturbed_modules)
    return float(np.mean(jaccards)), {
        gene_id: count / runs for gene_id, count in occurrences.items()
    }


@dataclass(frozen=True)
class EnrichmentIndex:
    """Pre-indexed annotation background for repeated module tests."""

    universe_size: int
    term_names: dict[str, str]
    term_gene_counts: dict[str, int]
    gene_terms: dict[str, tuple[str, ...]]


def build_enrichment_index(
    annotations: pd.DataFrame,
    universe_genes: set[str],
    *,
    term_column: str,
    name_column: str,
) -> EnrichmentIndex:
    """Build a reusable enrichment index over a fixed gene universe."""
    selected = annotations.loc[
        annotations["gene_id"].astype(str).isin(universe_genes),
        ["gene_id", term_column, name_column],
    ].drop_duplicates()
    term_names: dict[str, str] = {}
    term_genes: dict[str, set[str]] = defaultdict(set)
    gene_terms: dict[str, set[str]] = defaultdict(set)
    for row in selected.itertuples(index=False, name=None):
        gene_id, term_id, term_name = map(str, row)
        term_names[term_id] = term_name
        term_genes[term_id].add(gene_id)
        gene_terms[gene_id].add(term_id)
    return EnrichmentIndex(
        universe_size=len(universe_genes),
        term_names=term_names,
        term_gene_counts={term: len(genes) for term, genes in term_genes.items()},
        gene_terms={gene: tuple(sorted(terms)) for gene, terms in gene_terms.items()},
    )


def enrich_module_from_index(
    module_genes: Iterable[str],
    index: EnrichmentIndex,
    *,
    minimum_overlap: int = 2,
) -> pd.DataFrame:
    """Calculate module enrichment from a fixed pre-indexed background."""
    module = set(map(str, module_genes))
    overlap_genes: dict[str, set[str]] = defaultdict(set)
    for gene_id in module:
        for term_id in index.gene_terms.get(gene_id, ()):
            overlap_genes[term_id].add(gene_id)
    records = []
    for term_id, overlap in overlap_genes.items():
        if len(overlap) < minimum_overlap:
            continue
        p_value = float(hypergeom.sf(
            len(overlap) - 1,
            index.universe_size,
            index.term_gene_counts[term_id],
            len(module),
        ))
        records.append({
            "term_id": term_id,
            "term_name": index.term_names[term_id],
            "overlap_count": len(overlap),
            "module_size": len(module),
            "term_gene_count": index.term_gene_counts[term_id],
            "universe_size": index.universe_size,
            "p_value": p_value,
            "overlap_gene_ids": sorted(overlap, key=int),
        })
    frame = pd.DataFrame.from_records(records)
    if frame.empty:
        return pd.DataFrame(columns=[
            "term_id", "term_name", "overlap_count", "module_size",
            "term_gene_count", "universe_size", "p_value", "fdr_bh",
            "overlap_gene_ids",
        ])
    frame["fdr_bh"] = multipletests(frame["p_value"], method="fdr_bh")[1]
    return frame.sort_values(
        ["fdr_bh", "p_value", "term_id"], kind="stable"
    ).reset_index(drop=True)


def overrepresentation_enrichment(
    module_genes: Iterable[str],
    annotations: pd.DataFrame,
    universe_genes: set[str],
    *,
    term_column: str,
    name_column: str,
    minimum_overlap: int = 2,
) -> pd.DataFrame:
    """Calculate enrichment, building a one-use fixed-background index."""
    index = build_enrichment_index(
        annotations,
        universe_genes,
        term_column=term_column,
        name_column=name_column,
    )
    return enrich_module_from_index(
        module_genes, index, minimum_overlap=minimum_overlap
    )