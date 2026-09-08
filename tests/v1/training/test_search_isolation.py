"""Validation-only search isolation tests (Milestone 4)."""

from __future__ import annotations

import pytest

from phenotype_network_v1.training.search import AccessLog, successive_halving


def test_access_log_forbids_test_tags() -> None:
    log = AccessLog()
    log.record("validation_query_instances")
    with pytest.raises(ValueError):
        log.record("test_diseases.txt")
    with pytest.raises(ValueError):
        log.record("frozen_test_labels")


def test_access_log_records() -> None:
    log = AccessLog()
    log.record("validation")
    log.record("validation")
    assert log.to_dict() == {"accesses": ["validation"]}


def test_successive_halving_runs_and_logs() -> None:
    def objective(config, access_log):  # noqa: ANN001
        access_log.record("validation_only")
        return float(config["x"]) ** 2 + float(config["y"])

    result = successive_halving(
        space={"x": [-1.0, 0.0, 1.0], "y": [0.0, 1.0]},
        objective=objective,
        seed=1,
    )
    assert result.best_config == {"x": 0.0, "y": 0.0}
    assert result.best_value == 0.0
    assert result.access_log.to_dict() == {"accesses": ["validation_only"]}
    assert len(result.log) >= 2


def test_search_reproducible_with_same_seed() -> None:
    def objective(config, access_log):  # noqa: ANN001
        access_log.record("validation")
        return (float(config["a"]) - 0.3) ** 2

    first = successive_halving(space={"a": [0.1, 0.3, 0.9]}, objective=objective, seed=5)
    second = successive_halving(space={"a": [0.1, 0.3, 0.9]}, objective=objective, seed=5)
    assert first.best_config == second.best_config
    assert first.best_value == second.best_value
