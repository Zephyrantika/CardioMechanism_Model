"""Counterfactual fidelity tests (Milestone 6)."""

from __future__ import annotations

import pytest

from phenotype_network_v1.evaluation.explanation_fidelity import counterfactual_fidelity


def test_fidelity_positive_when_edge_matters() -> None:
    result = counterfactual_fidelity(
        path_nodes=[0, 1, 5],
        path_relations=["hpo_hpo", "hpo_gene"],
        score_gene=lambda: 0.9,
        score_gene_without_edge=lambda u, v: 0.4 if (u, v) == (1, 5) else 0.8,
    )
    assert result["baseline_score"] == 0.9
    assert result["fidelity"] == pytest.approx(0.1)


def test_relation_sequence_reported() -> None:
    result = counterfactual_fidelity(
        path_nodes=[2, 3],
        path_relations=["hpo_gene"],
        score_gene=lambda: 1.0,
        score_gene_without_edge=lambda u, v: 0.0,
    )
    assert result["edge_deletions"][0]["edge"] == [2, 3]
    assert result["mean_fidelity"] == 1.0


def test_single_node_path_zero_fidelity() -> None:
    result = counterfactual_fidelity(
        path_nodes=[7],
        path_relations=[],
        score_gene=lambda: 0.5,
        score_gene_without_edge=lambda u, v: 0.0,
    )
    assert result["fidelity"] == 0.0
