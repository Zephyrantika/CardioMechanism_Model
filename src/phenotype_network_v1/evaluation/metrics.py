"""Ranking metrics for V1 evaluation (Milestone 8).

Recall@k and MRR over the frozen candidate universe, comparable to the V0
evaluator (which stays the primary V0 comparator; V1 uses the same metric
definitions for direct comparison).
"""

from __future__ import annotations

from typing import Any, Sequence


def recall_at_k(ranked: Sequence[str], positives: set[str], k: int) -> float:
    if k <= 0:
        raise ValueError("k must be positive")
    if not positives:
        return float("nan")
    top = ranked[:k]
    return sum(gene in positives for gene in top) / len(positives)


def mean_reciprocal_rank(ranked: Sequence[str], positives: set[str]) -> float:
    """MRR over the ranked list; 0 when no positive appears."""
    if not positives:
        return float("nan")
    for index, gene in enumerate(ranked, start=1):
        if gene in positives:
            return 1.0 / index
    return 0.0


def compute_metrics(
    ranked_gene_ids: list[str],
    positive_gene_ids: list[str],
    recall_ks: Sequence[int] = (10, 20),
) -> dict[str, Any]:
    positives = set(positive_gene_ids)
    return {
        "recall_k10": recall_at_k(ranked_gene_ids, positives, 10),
        "recall_k20": recall_at_k(ranked_gene_ids, positives, 20),
        "mrr": mean_reciprocal_rank(ranked_gene_ids, positives),
        "n_positive": len(positives),
        "n_ranked": len(ranked_gene_ids),
    }


__all__ = ["compute_metrics", "mean_reciprocal_rank", "recall_at_k"]
