"""Robustness evaluation for frozen V1 checkpoints (Milestone 7).

Preregistered structural perturbations (random edge deletion with an explicit
seed) re-score a query and measure rank stability against the unperturbed
scores. Perturbation never retrains or changes model weights.
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np


def perturb_edge_index(
    edge_index: Any,
    drop_fraction: float,
    *,
    seed: int,
    edge_type: Any | None = None,
) -> Any:
    """Return edge_index (and edge_type when given) minus dropped edges."""
    import torch

    if not 0.0 <= drop_fraction < 1.0:
        raise ValueError("drop_fraction must be in [0, 1)")
    rng = np.random.RandomState(seed)
    n = edge_index.size(1)
    keep = rng.rand(n) >= drop_fraction
    keep_tensor = torch.tensor(keep, dtype=torch.bool)
    filtered = edge_index[:, keep_tensor]
    if edge_type is None:
        return filtered
    return filtered, edge_type[keep_tensor]


def rank_stability(
    scores_a: list[float],
    scores_b: list[float],
) -> float | None:
    """Spearman correlation between two score lists over the same genes."""
    if len(scores_a) != len(scores_b) or len(scores_a) < 3:
        return None
    array_a = np.asarray(scores_a, dtype=float)
    array_b = np.asarray(scores_b, dtype=float)
    if float(array_a.std()) == 0.0 or float(array_b.std()) == 0.0:
        return None
    rank_a = np.argsort(np.argsort(-array_a))
    rank_b = np.argsort(np.argsort(-array_b))
    n = len(rank_a)
    cov = ((rank_a - rank_a.mean()) * (rank_b - rank_b.mean())).sum()
    denom = np.sqrt(
        ((rank_a - rank_a.mean()) ** 2).sum() * ((rank_b - rank_b.mean()) ** 2).sum()
    )
    return float(cov / denom) if denom > 0 else None


def evaluate_perturbation_robustness(
    *,
    score_fn: Callable[[Any], list[float]],
    perturbed_score_fn: Callable[[int], list[float]],
    seeds: list[int],
) -> dict[str, Any]:
    """Rank stability of one query across preregistered perturbations."""
    baseline = score_fn()
    stabilities: list[float | None] = []
    for seed in seeds:
        perturbed = perturbed_score_fn(seed)
        stabilities.append(rank_stability(baseline, perturbed))
    valid = [value for value in stabilities if value is not None]
    return {
        "baseline_len": len(baseline),
        "seeds": seeds,
        "stabilities": stabilities,
        "mean_stability": float(np.mean(valid)) if valid else None,
    }


__all__ = [
    "evaluate_perturbation_robustness",
    "perturb_edge_index",
    "rank_stability",
]
