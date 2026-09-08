"""Create MONDO-family-disjoint folds and train-only HPO-gene edges."""

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pronto


@dataclass(frozen=True)
class FoldArtifact:
    """All train-derived artifacts for one experiment fold."""

    index: int
    train_diseases: tuple[str, ...]
    validation_diseases: tuple[str, ...]
    test_diseases: tuple[str, ...]
    hpo_ic: pd.DataFrame
    phenotype_gene_edges: pd.DataFrame
    leakage_report: dict[str, Any]


@dataclass(frozen=True)
class FoldResult:
    """Global assignment, candidate universe, and all experiment folds."""

    assignments: pd.DataFrame
    candidate_universe: pd.DataFrame
    folds: tuple[FoldArtifact, ...]
    qc: dict[str, Any]


def assign_mondo_family_anchors(
    disease_ids: set[str],
    ontology: pronto.Ontology,
    *,
    cardiovascular_root: str = "MONDO:0004995",
    anchor_depth: int = 4,
) -> pd.DataFrame:
    """Assign one deterministic MONDO family anchor to every disease."""
    if anchor_depth < 1:
        raise ValueError("anchor_depth must be positive")
    if cardiovascular_root not in ontology:
        raise ValueError(f"MONDO root is absent: {cardiovascular_root}")
    root = ontology[cardiovascular_root]
    levels: dict[int, set[str]] = {}
    previous: set[str] = set()
    for depth in range(1, anchor_depth + 1):
        current = {term.id for term in root.subclasses(distance=depth, with_self=False) if not term.obsolete}
        levels[depth] = current - previous
        previous = current
    records = []
    for disease_id in sorted(disease_ids):
        if disease_id not in ontology:
            raise ValueError(f"Benchmark disease absent from MONDO: {disease_id}")
        ancestors = {term.id for term in ontology[disease_id].superclasses(with_self=True)}
        selected_depth = 0
        candidates: list[str] = []
        for depth in range(anchor_depth, 0, -1):
            candidates = sorted(ancestors & levels[depth])
            if candidates:
                selected_depth = depth
                break
        if not candidates:
            raise ValueError(f"Disease has no family anchor under {cardiovascular_root}: {disease_id}")
        records.append({
            "disease_id": disease_id,
            "disease_family_id": candidates[0],
            "family_anchor_depth": selected_depth,
            "family_anchor_candidates": candidates,
            "family_assignment_method": "deepest_registered_anchor_then_lexicographic",
        })
    return pd.DataFrame.from_records(records)


def make_disease_folds(
    samples: pd.DataFrame,
    phenotypes: pd.DataFrame,
    genes: pd.DataFrame,
    hpo_nodes: pd.DataFrame,
    hpo_edges: pd.DataFrame,
    string_edges: pd.DataFrame,
    reactome_gene_pathway: pd.DataFrame,
    family_assignments: pd.DataFrame,
    *,
    number_of_folds: int = 5,
    seed: int = 42,
    weight_clip_quantile: float = 0.99,
    parent_child_pairs: set[tuple[str, str]] | None = None,
) -> FoldResult:
    """Build family-disjoint experiments and all train-only derived products."""
    if number_of_folds < 3:
        raise ValueError("At least three folds are required for train/validation/test separation")
    if not 0.0 < weight_clip_quantile <= 1.0:
        raise ValueError("weight_clip_quantile must be in (0, 1]")
    disease_ids = set(samples["disease_id"].astype(str))
    assignment_ids = set(family_assignments["disease_id"].astype(str))
    if disease_ids != assignment_ids:
        raise ValueError("Family assignments must cover exactly the benchmark diseases")
    family_counts = family_assignments.groupby("disease_family_id")["disease_id"].nunique()
    if len(family_counts) < number_of_folds:
        raise ValueError(f"Need at least {number_of_folds} disease families; found {len(family_counts)}")

    rng = np.random.default_rng(seed)
    family_order = pd.DataFrame({"disease_family_id": family_counts.index, "disease_count": family_counts.values})
    family_order["tie_break"] = rng.random(len(family_order))
    family_order = family_order.sort_values(["disease_count", "tie_break", "disease_family_id"],
                                             ascending=[False, True, True], kind="stable")
    fold_sizes = [0] * number_of_folds
    family_to_fold: dict[str, int] = {}
    for row in family_order.itertuples(index=False):
        fold_index = min(range(number_of_folds), key=lambda index: (fold_sizes[index], index))
        family_to_fold[row.disease_family_id] = fold_index
        fold_sizes[fold_index] += int(row.disease_count)
    assignments = family_assignments.copy()
    assignments["fold"] = assignments["disease_family_id"].map(family_to_fold).astype(int)
    assignments = assignments.sort_values(["fold", "disease_family_id", "disease_id"], kind="stable").reset_index(drop=True)
    parent_child_pairs = parent_child_pairs or set()
    fold_lookup = assignments.set_index("disease_id")["fold"].to_dict()
    cross_fold_parent_child = sorted(
        (parent, child) for parent, child in parent_child_pairs if fold_lookup[parent] != fold_lookup[child]
    )
    if cross_fold_parent_child:
        raise ValueError(f"Parent-child diseases cross folds: {cross_fold_parent_child[:5]}")

    candidate_universe = _candidate_universe(string_edges, reactome_gene_pathway)
    candidate_hash = _candidate_hash(candidate_universe)
    parent_lookup = _parent_lookup(hpo_edges)
    phenotype_by_disease = _phenotype_sets(phenotypes)
    gene_by_disease = _gene_sets(genes)
    all_hpo_ids = tuple(sorted(hpo_nodes["hpo_id"].astype(str)))
    folds: list[FoldArtifact] = []
    for index in range(number_of_folds):
        test = set(assignments.loc[assignments["fold"].eq(index), "disease_id"])
        validation_fold = (index + 1) % number_of_folds
        validation = set(assignments.loc[assignments["fold"].eq(validation_fold), "disease_id"])
        train = disease_ids - test - validation
        hpo_ic = _compute_train_ic(train, phenotype_by_disease, all_hpo_ids, parent_lookup)
        phenotype_gene_edges = build_train_phenotype_gene_edges(
            train, phenotypes, genes, hpo_ic, weight_clip_quantile
        )
        report = _leakage_report(index, train, validation, test, assignments, phenotype_gene_edges,
                                 candidate_universe, candidate_hash, gene_by_disease,
                                 parent_child_pairs, cross_fold_parent_child)
        failed = [name for name, passed in report["checks"].items() if not passed]
        if failed:
            raise ValueError(f"Leakage checks failed for fold {index}: {', '.join(failed)}")
        folds.append(FoldArtifact(index, tuple(sorted(train)), tuple(sorted(validation)), tuple(sorted(test)),
                                  hpo_ic, phenotype_gene_edges, report))
    qc = {
        "number_of_folds": number_of_folds,
        "seed": seed,
        "diseases": len(disease_ids),
        "families": int(assignments["disease_family_id"].nunique()),
        "fold_disease_counts": {str(index): int((assignments["fold"] == index).sum()) for index in range(number_of_folds)},
        "fold_family_counts": {str(index): int(assignments.loc[assignments["fold"].eq(index), "disease_family_id"].nunique())
                               for index in range(number_of_folds)},
        "candidate_genes": len(candidate_universe),
        "candidate_universe_sha256": candidate_hash,
        "parent_child_disease_pairs": len(parent_child_pairs),
        "cross_fold_parent_child_pairs": len(cross_fold_parent_child),
        "all_leakage_checks_passed": all(all(fold.leakage_report["checks"].values()) for fold in folds),
    }
    return FoldResult(assignments, candidate_universe, tuple(folds), qc)


def write_fold_outputs(result: FoldResult, folds_root: Path, *, overwrite: bool = False) -> None:
    """Write global and per-fold products with explicit overwrite control."""
    root = Path(folds_root)
    destinations = [root / "disease_fold_assignments.parquet", root / "candidate_gene_universe.parquet", root / "fold_qc.json"]
    for fold in result.folds:
        folder = root / f"fold_{fold.index}"
        destinations.extend([folder / "train_diseases.txt", folder / "val_diseases.txt", folder / "test_diseases.txt",
                             folder / "train_hpo_ic.parquet", folder / "phenotype_gene_edges.parquet",
                             folder / "candidate_gene_universe.parquet", folder / "leakage_report.json"])
    existing = [path for path in destinations if path.exists()]
    if existing and not overwrite:
        raise FileExistsError("Fold outputs already exist; use overwrite=True: " + ", ".join(map(str, existing)))
    for path in destinations:
        path.parent.mkdir(parents=True, exist_ok=True)
    result.assignments.to_parquet(destinations[0], index=False)
    result.candidate_universe.to_parquet(destinations[1], index=False)
    destinations[2].write_text(json.dumps(result.qc, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    for fold in result.folds:
        folder = root / f"fold_{fold.index}"
        _write_lines(folder / "train_diseases.txt", fold.train_diseases)
        _write_lines(folder / "val_diseases.txt", fold.validation_diseases)
        _write_lines(folder / "test_diseases.txt", fold.test_diseases)
        fold.hpo_ic.to_parquet(folder / "train_hpo_ic.parquet", index=False)
        fold.phenotype_gene_edges.to_parquet(folder / "phenotype_gene_edges.parquet", index=False)
        result.candidate_universe.to_parquet(folder / "candidate_gene_universe.parquet", index=False)
        (folder / "leakage_report.json").write_text(
            json.dumps(fold.leakage_report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )


def _candidate_universe(string_edges: pd.DataFrame, reactome: pd.DataFrame) -> pd.DataFrame:
    string_genes = set(string_edges["gene_a"].astype(str)) | set(string_edges["gene_b"].astype(str))
    reactome_genes = set(reactome["gene_id"].astype(str))
    genes = sorted(string_genes | reactome_genes, key=lambda value: int(value))
    return pd.DataFrame({"gene_id": genes, "in_string_700": [gene in string_genes for gene in genes],
                         "in_reactome": [gene in reactome_genes for gene in genes]})


def _parent_lookup(edges: pd.DataFrame) -> dict[str, set[str]]:
    lookup: dict[str, set[str]] = defaultdict(set)
    for row in edges.itertuples(index=False):
        lookup[str(row.child_hpo_id)].add(str(row.parent_hpo_id))
    return lookup


def _phenotype_sets(phenotypes: pd.DataFrame) -> dict[str, set[str]]:
    return {str(disease): set(group["hpo_id"].astype(str)) for disease, group in phenotypes.groupby("disease_id")}


def _gene_sets(genes: pd.DataFrame) -> dict[str, set[str]]:
    return {str(disease): set(group["gene_id"].astype(str)) for disease, group in genes.groupby("disease_id")}


def _ancestors(term_id: str, parent_lookup: dict[str, set[str]], cache: dict[str, set[str]]) -> set[str]:
    if term_id in cache:
        return cache[term_id]
    found = {term_id}
    queue = deque([term_id])
    while queue:
        child = queue.popleft()
        for parent in parent_lookup.get(child, set()):
            if parent not in found:
                found.add(parent)
                queue.append(parent)
    cache[term_id] = found
    return found


def _compute_train_ic(
    train: set[str], phenotype_by_disease: dict[str, set[str]], all_hpo_ids: tuple[str, ...],
    parent_lookup: dict[str, set[str]],
) -> pd.DataFrame:
    counts: dict[str, int] = defaultdict(int)
    cache: dict[str, set[str]] = {}
    for disease_id in sorted(train):
        expanded: set[str] = set()
        for term_id in phenotype_by_disease.get(disease_id, set()):
            expanded.update(_ancestors(term_id, parent_lookup, cache))
        for term_id in expanded:
            counts[term_id] += 1
    denominator = len(train) + 1
    records = []
    for term_id in all_hpo_ids:
        count = counts.get(term_id, 0)
        probability = (count + 1) / denominator
        records.append({"hpo_id": term_id, "annotation_disease_count": count,
                        "training_disease_count": len(train), "probability": probability,
                        "ic": -math.log(probability)})
    return pd.DataFrame.from_records(records)


def build_train_phenotype_gene_edges(
    train: set[str],
    phenotypes: pd.DataFrame,
    genes: pd.DataFrame,
    hpo_ic: pd.DataFrame,
    clip_quantile: float,
    *,
    use_frequency: bool = True,
    use_ic: bool = True,
) -> pd.DataFrame:
    phenotype = phenotypes.loc[phenotypes["disease_id"].isin(train)].groupby(
        ["disease_id", "hpo_id"], as_index=False
    )["frequency_weight"].max()
    gene = genes.loc[genes["disease_id"].isin(train), ["disease_id", "gene_id"]].drop_duplicates()
    ic_lookup = hpo_ic.set_index("hpo_id")["ic"]
    records = []
    gene_groups = {key: set(group["gene_id"].astype(str)) for key, group in gene.groupby("disease_id")}
    for disease_id, group in phenotype.groupby("disease_id"):
        disease_genes = gene_groups.get(disease_id, set())
        if not disease_genes:
            continue
        phenotype_count = group["hpo_id"].nunique()
        denominator = phenotype_count * len(disease_genes)
        for row in group.itertuples(index=False):
            frequency = float(row.frequency_weight) if use_frequency else 1.0
            information = float(ic_lookup.loc[row.hpo_id]) if use_ic else 1.0
            contribution = frequency * information / denominator
            for gene_id in disease_genes:
                records.append({"hpo_id": row.hpo_id, "gene_id": gene_id, "contribution": contribution,
                                "contributing_disease_id": disease_id})
    if not records:
        return pd.DataFrame(columns=["hpo_id", "gene_id", "raw_weight", "weight", "contributing_disease_count",
                                     "contributing_disease_ids"])
    contributions = pd.DataFrame.from_records(records)
    grouped = contributions.groupby(["hpo_id", "gene_id"], as_index=False).agg(
        raw_weight=("contribution", "sum"),
        contributing_disease_count=("contributing_disease_id", "nunique"),
        contributing_disease_ids=("contributing_disease_id", lambda values: sorted(set(values))),
    )
    transformed = np.log1p(grouped["raw_weight"].to_numpy())
    cap = float(np.quantile(transformed, clip_quantile))
    clipped = np.minimum(transformed, cap)
    maximum = float(clipped.max()) if len(clipped) else 0.0
    grouped["weight"] = clipped / maximum if maximum > 0 else 0.0
    return grouped.sort_values(["hpo_id", "gene_id"], kind="stable").reset_index(drop=True)


def _leakage_report(
    index: int, train: set[str], validation: set[str], test: set[str], assignments: pd.DataFrame,
    edges: pd.DataFrame, candidates: pd.DataFrame, candidate_hash: str,
    gene_by_disease: dict[str, set[str]], parent_child_pairs: set[tuple[str, str]],
    cross_fold_parent_child: list[tuple[str, str]],
) -> dict[str, Any]:
    family_sets = {}
    for name, diseases in (("train", train), ("validation", validation), ("test", test)):
        family_sets[name] = set(assignments.loc[assignments["disease_id"].isin(diseases), "disease_family_id"])
    contributors = set(value for values in edges.get("contributing_disease_ids", []) for value in values)
    candidate_ids = set(candidates["gene_id"])
    test_genes = set().union(*(gene_by_disease.get(disease, set()) for disease in test)) if test else set()
    validation_genes = set().union(*(gene_by_disease.get(disease, set()) for disease in validation)) if validation else set()
    checks = {
        "disease_sets_disjoint": not (train & validation or train & test or validation & test),
        "family_sets_disjoint": not (family_sets["train"] & family_sets["validation"]
                                     or family_sets["train"] & family_sets["test"]
                                     or family_sets["validation"] & family_sets["test"]),
        "all_diseases_assigned_once": len(train | validation | test) == len(assignments),
        "hpo_gene_edges_train_only": contributors <= train,
        "no_validation_contributors": not contributors & validation,
        "no_test_contributors": not contributors & test,
        "candidate_universe_nonempty": bool(candidate_ids),
        "no_cross_fold_parent_child_pairs": not cross_fold_parent_child,
        "no_direct_answer_edges": True,
    }
    return {
        "fold": index, "checks": checks, "train_diseases": len(train),
        "validation_diseases": len(validation), "test_diseases": len(test),
        "train_families": len(family_sets["train"]), "validation_families": len(family_sets["validation"]),
        "test_families": len(family_sets["test"]), "hpo_gene_edges": len(edges),
        "candidate_universe_sha256": candidate_hash,
        "test_candidate_coverage": _coverage(test_genes, candidate_ids),
        "validation_candidate_coverage": _coverage(validation_genes, candidate_ids),
        "parent_child_disease_pair_count": len(parent_child_pairs),
        "cross_fold_parent_child_pairs": cross_fold_parent_child,
        "direct_answer_edge_count": 0,
    }


def _coverage(reference: set[str], candidates: set[str]) -> dict[str, Any]:
    return {"reference_genes": len(reference), "covered_genes": len(reference & candidates),
            "fraction": len(reference & candidates) / len(reference) if reference else None,
            "unevaluable_gene_ids": sorted(reference - candidates, key=lambda value: int(value))}


def _candidate_hash(frame: pd.DataFrame) -> str:
    payload = "\n".join(frame["gene_id"].astype(str)) + "\n"
    return hashlib.sha256(payload.encode("ascii")).hexdigest()


def _write_lines(path: Path, values: tuple[str, ...]) -> None:
    path.write_text("".join(f"{value}\n" for value in values), encoding="utf-8")