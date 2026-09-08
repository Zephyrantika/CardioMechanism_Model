"""Degree-bias evaluation for V1 gene rankings (Milestone 4).

CoLiPE-style degree-bias evaluation (design reuse only; no upstream code):
report Spearman correlation between gene scores/ranks and graph degree,
before and after calibration, plus an explicit zero-variance audit. Gene
degree is computed from the frozen fold graph and is never re-derived from
test labels.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np


def _spearman(a: Sequence[float], b: Sequence[float]) -> float | None:
    array_a = np.asarray(a, dtype=float)
    array_b = np.asarray(b, dtype=float)
    if len(array_a) < 3 or float(array_a.std()) == 0.0 or float(array_b.std()) == 0.0:
        return None
    rank_a = np.argsort(np.argsort(array_a))
    rank_b = np.argsort(np.argsort(array_b))
    n = len(rank_a)
    mean_a = rank_a.mean()
    mean_b = rank_b.mean()
    cov = ((rank_a - mean_a) * (rank_b - mean_b)).sum()
    denom = np.sqrt(
        ((rank_a - mean_a) ** 2).sum() * ((rank_b - mean_b) ** 2).sum()
    )
    if denom == 0:
        return None
    return float(cov / denom)


def compute_degree_bias(
    gene_scores: Sequence[float],
    gene_degrees: Sequence[int],
) -> dict[str, Any]:
    """Return degree-bias statistics for one query ranking.

    ``gene_degrees`` are out/in degrees of the candidate genes in the frozen
    fold graph. Returns Spearman correlations and a zero-variance audit.
    """
    scores = np.asarray(list(gene_scores), dtype=float)
    degrees = np.asarray(list(gene_degrees), dtype=float)
    if scores.shape != degrees.shape:
        raise ValueError("gene_scores and gene_degrees must have equal length")
    audit = {
        "zero_variance_scores": bool(float(scores.std()) == 0.0),
        "zero_variance_degrees": bool(float(degrees.std()) == 0.0),
        "n_genes": int(len(scores)),
    }
    return {
        "spearman_score_vs_degree": _spearman(scores.tolist(), degrees.tolist()),
        "spearman_rank_vs_degree": _spearman(
            np.argsort(-scores).tolist(), degrees.tolist()
        ),
        "zero_variance_audit": audit,
    }


__all__ = ["compute_degree_bias"]
