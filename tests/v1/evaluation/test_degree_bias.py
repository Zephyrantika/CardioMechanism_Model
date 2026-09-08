"""Degree-bias evaluation tests (Milestone 4)."""

from __future__ import annotations

import numpy as np
import pytest

from phenotype_network_v1.evaluation.bias import compute_degree_bias


def test_positive_score_degree_correlation_detected() -> None:
    degrees = [float(x) for x in range(1, 51)]
    scores = [float(2 * x) for x in degrees]
    report = compute_degree_bias(scores, degrees)
    assert report["spearman_score_vs_degree"] is not None
    assert report["spearman_score_vs_degree"] > 0.9


def test_zero_variance_audit() -> None:
    report = compute_degree_bias([1.0, 1.0, 1.0, 1.0], [1, 2, 3, 4])
    assert report["zero_variance_audit"]["zero_variance_scores"] is True
    assert report["spearman_score_vs_degree"] is None


def test_mismatched_lengths_rejected() -> None:
    with pytest.raises(ValueError):
        compute_degree_bias([1.0, 2.0], [1, 2, 3])


def test_report_schema() -> None:
    rng = np.random.RandomState(0)
    scores = [float(x) for x in rng.rand(50)]
    degrees = [int(x) for x in rng.randint(1, 100, size=50)]
    report = compute_degree_bias(scores, degrees)
    assert "spearman_score_vs_degree" in report
    assert "spearman_rank_vs_degree" in report
    assert report["zero_variance_audit"]["n_genes"] == 50
