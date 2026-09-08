"""Deterministic sampling utilities for V1 training (Milestone 2).

Every randomised operation derives an explicit seed from a base seed plus an
index so runs are reproducible and auditable. Milestone 4 adds degree-
stratified bagging on top of these primitives.
"""

from __future__ import annotations

import random
from typing import Iterable, Sequence

import numpy as np


def derive_seed(base_seed: int, index: int) -> int:
    """Derive a reproducible per-member seed."""
    if not isinstance(base_seed, int) or not isinstance(index, int):
        raise TypeError("base_seed and index must be ints")
    if index < 0:
        raise ValueError("index must be non-negative")
    return (base_seed * 1000003 + index * 7919) % (2**31 - 1)


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch RNGs deterministically."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
    except ImportError:  # pragma: no cover - torch optional on CPU scaffold
        pass


def sample_without_replacement(
    population: Sequence[int],
    k: int,
    *,
    seed: int,
    rng: random.Random | None = None,
) -> list[int]:
    """Deterministically sample ``k`` indices without replacement."""
    if k < 0 or k > len(population):
        raise ValueError("k must satisfy 0 <= k <= len(population)")
    sampler = rng or random.Random(derive_seed(seed, 0))
    return sorted(sampler.sample(population, k))


def stratified_sample_indices(
    groups: dict[str, Sequence[int]],
    k_per_group: int,
    *,
    seed: int,
) -> dict[str, list[int]]:
    """Sample ``k_per_group`` indices per group with an explicit seed."""
    sampler = random.Random(derive_seed(seed, 1))
    return {
        name: sample_without_replacement(
            list(items), min(k_per_group, len(items)), seed=seed, rng=sampler
        )
        for name, items in sorted(groups.items())
    }


def iter_epoch_batches(
    indices: Sequence[int],
    batch_size: int,
    *,
    seed: int,
    shuffle: bool = True,
) -> Iterable[list[int]]:
    """Yield deterministic epoch batches over ``indices``."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    order = list(indices)
    if shuffle:
        rng = random.Random(derive_seed(seed, 2))
        rng.shuffle(order)
    for start in range(0, len(order), batch_size):
        yield order[start : start + batch_size]


__all__ = [
    "derive_seed",
    "iter_epoch_batches",
    "sample_without_replacement",
    "seed_everything",
    "stratified_sample_indices",
]
