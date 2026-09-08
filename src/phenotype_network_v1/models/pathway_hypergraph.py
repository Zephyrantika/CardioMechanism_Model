"""Query-specific Reactome pathway hypergraph context (V1 Milestone 5).

Builds query-specific pathway activations from frozen Reactome membership:
for a query's ranked gene scores, a pathway is activated by the genes it
contains. Membership comes only from frozen Reactome evidence
(``reactome_gene_pathway.parquet``, primary-graph pathways) and is never
derived from test labels.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


def load_reactome_membership(path: str | Path) -> dict[str, set[str]]:
    """Load frozen gene -> pathway membership restricted to primary pathways."""
    edges = pd.read_parquet(path)
    membership: dict[str, set[str]] = {}
    for gene_id, pathway_id in edges[["gene_id", "pathway_id"]].itertuples(index=False):
        membership.setdefault(str(pathway_id), set()).add(str(gene_id))
    return membership


def pathway_activation(
    membership: dict[str, set[str]],
    gene_ids: list[str],
    gene_scores: list[float],
) -> dict[str, float]:
    """Sum gene scores per pathway for one query ranking.

    ``gene_ids``/``gene_scores`` must be aligned; scores of genes outside any
    pathway simply do not contribute. Returns {pathway_id: activation}.
    """
    if len(gene_ids) != len(gene_scores):
        raise ValueError("gene_ids and gene_scores must be aligned")
    activation: dict[str, float] = {}
    score_by_gene = dict(zip(gene_ids, gene_scores))
    for pathway_id, pathway_genes in membership.items():
        total = 0.0
        for gene_id in pathway_genes:
            score = score_by_gene.get(gene_id)
            if score is not None:
                total += float(score)
        if total > 0.0:
            activation[pathway_id] = total
    return activation


def module_top_genes(
    membership: dict[str, set[str]],
    pathway_id: str,
    gene_ids: list[str],
    gene_scores: list[float],
    *,
    max_genes: int,
) -> list[str]:
    """Return the top-scoring candidate genes of a pathway (module members)."""
    pathway_genes = membership.get(pathway_id, set())
    scored = [
        (float(score), str(gene_id))
        for gene_id, score in zip(gene_ids, gene_scores)
        if str(gene_id) in pathway_genes
    ]
    scored.sort(reverse=True)
    return [gene_id for _, gene_id in scored[:max_genes]]


def selected_queries(
    query_frame: pd.DataFrame, split: str = "test", limit: int | None = None
) -> pd.DataFrame:
    frame = query_frame[query_frame["split"] == split]
    if limit:
        frame = frame.head(limit)
    return frame.reset_index(drop=True)


__all__ = [
    "load_reactome_membership",
    "module_top_genes",
    "pathway_activation",
    "selected_queries",
]
