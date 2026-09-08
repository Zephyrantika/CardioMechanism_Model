import random

import numpy as np
import pytest

from phenotype_network_v0.random_utils import set_random_seed


def draw_values(seed: int) -> tuple[float, float]:
    set_random_seed(seed)
    return random.random(), float(np.random.random())


def test_same_seed_is_deterministic() -> None:
    assert draw_values(42) == draw_values(42)


def test_different_seeds_change_values() -> None:
    assert draw_values(1) != draw_values(2)


@pytest.mark.parametrize("seed", [1.5, "42", True, None])
def test_invalid_seed_type_fails(seed: object) -> None:
    with pytest.raises(TypeError, match="integer"):
        set_random_seed(seed)  # type: ignore[arg-type]

