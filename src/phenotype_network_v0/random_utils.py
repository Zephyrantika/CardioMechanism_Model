"""Randomness controls for reproducible experiments."""

from __future__ import annotations

import random

import numpy as np


def set_random_seed(seed: int) -> None:
    """Seed Python and NumPy random number generators."""
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an integer")
    if not 0 <= seed <= 2**32 - 1:
        raise ValueError("seed must be between 0 and 2**32 - 1")
    random.seed(seed)
    np.random.seed(seed)

