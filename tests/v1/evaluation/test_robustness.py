"""Robustness perturbation tests (Milestone 7)."""

from __future__ import annotations

import torch

from phenotype_network_v1.evaluation.robustness import (
    evaluate_perturbation_robustness,
    perturb_edge_index,
    rank_stability,
)


def test_perturb_drops_edges() -> None:
    edge_index = torch.tensor(
        [[0] * 100, list(range(1, 101))], dtype=torch.long
    )
    perturbed = perturb_edge_index(edge_index, 0.5, seed=0)
    assert perturbed.size(1) < edge_index.size(1)


def test_perturb_deterministic() -> None:
    edge_index = torch.tensor([[0] * 100, list(range(1, 101))], dtype=torch.long)
    a = perturb_edge_index(edge_index, 0.3, seed=7)
    b = perturb_edge_index(edge_index, 0.3, seed=7)
    assert torch.equal(a, b)


def test_rank_stability_perfect_positive() -> None:
    assert rank_stability([1.0, 2.0, 3.0], [10.0, 20.0, 30.0]) == 1.0


def test_rank_stability_constant_inputs_none() -> None:
    assert rank_stability([1.0, 1.0, 1.0], [1.0, 2.0, 3.0]) is None


def test_perturbation_robustness_runs() -> None:
    result = evaluate_perturbation_robustness(
        score_fn=lambda: [float(x) for x in torch.rand(10).tolist()],
        perturbed_score_fn=lambda seed: [float(x + 0.01) for x in torch.rand(10).tolist()],
        seeds=[1],
    )
    assert result["baseline_len"] == 10
