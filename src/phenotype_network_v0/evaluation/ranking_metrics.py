"""Disease-gene ranking metrics with explicit candidate coverage."""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np


def deterministic_score_order(
    gene_ids: Iterable[str],
    scores: Iterable[float],
    *,
    descending: bool = True,
) -> tuple[str, ...]:
    """Order candidate genes by score with numeric NCBI Gene ID tie breaking."""
    genes = np.asarray([str(value) for value in gene_ids], dtype=object)
    values = np.asarray(list(scores), dtype=float)
    if len(genes) != len(values):
        raise ValueError("Gene and score lengths differ")
    if len(set(genes)) != len(genes):
        raise ValueError("Candidate gene identifiers must be unique")
    if not np.isfinite(values).all():
        raise ValueError("Ranking scores must be finite")
    numbers = np.asarray([int(value) for value in genes], dtype=np.int64)
    primary = -values if descending else values
    return tuple(genes[np.lexsort((numbers, primary))])


def evaluate_positive_ranks(
    positive_ranks: Iterable[int],
    *,
    candidate_gene_count: int,
    reference_positive_count: int,
    recall_ks: tuple[int, ...] = (1, 5, 10, 20),
) -> dict[str, float | int | None]:
    """Evaluate known positive rank positions, including uncovered reference genes."""
    if candidate_gene_count < 0 or reference_positive_count < 0:
        raise ValueError("Candidate and reference counts must be nonnegative")
    if any(k < 1 for k in recall_ks):
        raise ValueError("Recall cutoffs must be positive")
    ranks = np.asarray(sorted(int(value) for value in positive_ranks), dtype=np.int64)
    if len(ranks) and (
        ranks.min() < 1
        or ranks.max() > candidate_gene_count
        or len(np.unique(ranks)) != len(ranks)
    ):
        raise ValueError("Positive ranks must be unique positions within the candidate list")
    if len(ranks) > reference_positive_count:
        raise ValueError("Covered positives exceed the reference count")
    result: dict[str, float | int | None] = {
        **{
            f"recall_at_{k}": (
                float(np.sum(ranks <= k) / reference_positive_count)
                if reference_positive_count else None
            )
            for k in recall_ks
        },
        "mrr": None,
        "average_precision": None,
        "auprc": None,
        "best_positive_rank": int(ranks[0]) if len(ranks) else None,
    }
    if reference_positive_count == 0:
        return result
    if not len(ranks):
        result.update({"mrr": 0.0, "average_precision": 0.0, "auprc": 0.0})
        return result
    positive_number = np.arange(1, len(ranks) + 1, dtype=float)
    precision_at_positive = positive_number / ranks
    previous_precision = np.divide(
        positive_number - 1.0,
        ranks - 1,
        out=np.ones(len(ranks), dtype=float),
        where=ranks > 1,
    )
    result["mrr"] = float(1.0 / ranks[0])
    result["average_precision"] = float(
        precision_at_positive.sum() / reference_positive_count
    )
    result["auprc"] = float(
        np.sum(0.5 * (precision_at_positive + previous_precision))
        / reference_positive_count
    )
    return result


def evaluate_ordered_ranking(
    ordered_gene_ids: Iterable[str],
    positive_gene_ids: set[str],
    *,
    recall_ks: tuple[int, ...] = (1, 5, 10, 20),
) -> dict[str, Any]:
    """Evaluate one complete or partial ranking without dropping uncovered positives."""
    genes = tuple(str(value) for value in ordered_gene_ids)
    positives = {str(value) for value in positive_gene_ids}
    if len(set(genes)) != len(genes):
        raise ValueError("Ranked candidate gene identifiers must be unique")
    if any(k < 1 for k in recall_ks):
        raise ValueError("Recall cutoffs must be positive")
    candidate_set = set(genes)
    covered = positives & candidate_set
    unresolved = sorted(positives - candidate_set, key=int)
    reference_count = len(positives)
    covered_count = len(covered)
    result: dict[str, Any] = {
        "candidate_gene_count": len(genes),
        "reference_positive_count": reference_count,
        "covered_positive_count": covered_count,
        "candidate_coverage": covered_count / reference_count if reference_count else None,
        "unevaluable_gene_ids": unresolved,
        "evaluable": covered_count > 0,
    }
    positive_positions = [
        index
        for index, gene in enumerate(genes, start=1)
        if gene in positives
    ]
    result.update(evaluate_positive_ranks(
        positive_positions,
        candidate_gene_count=len(genes),
        reference_positive_count=reference_count,
        recall_ks=recall_ks,
    ))
    return result
