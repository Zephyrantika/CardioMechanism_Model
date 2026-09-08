"""Statistical comparisons for V1 evaluation (Milestone 8).

Paired, seed-based comparison of two model rankings over the same queries
(e.g. V1 model vs V0 raw RWR) using a bootstrap over queries. Results are
reported with explicit caveats; a passing engineering pipeline is not a
positive scientific claim.
"""

from __future__ import annotations

import numpy as np


def paired_bootstrap_delta(
    first: list[float],
    second: list[float],
    *,
    seed: int,
    iterations: int = 1000,
) -> dict[str, float]:
    """Mean (first - second) delta with bootstrap confidence interval."""
    if len(first) != len(second) or len(first) == 0:
        raise ValueError("first and second must be paired and non-empty")
    first_array = np.asarray(first, dtype=float)
    second_array = np.asarray(second, dtype=float)
    deltas = first_array - second_array
    rng = np.random.RandomState(seed)
    sampled = []
    for _ in range(iterations):
        index = rng.randint(0, len(deltas), size=len(deltas))
        sampled.append(float(deltas[index].mean()))
    sampled = np.asarray(sampled)
    return {
        "observed_delta": float(deltas.mean()),
        "ci_low": float(np.percentile(sampled, 2.5)),
        "ci_high": float(np.percentile(sampled, 97.5)),
        "iterations": int(iterations),
        "n_queries": int(len(deltas)),
        "seed": int(seed),
    }


__all__ = ["paired_bootstrap_delta"]
