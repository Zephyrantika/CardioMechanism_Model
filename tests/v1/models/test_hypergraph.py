"""Pathway hypergraph and module-loss tests (Milestone 5)."""

from __future__ import annotations

from phenotype_network_v1.models.module_losses import module_regularization_loss
from phenotype_network_v1.models.pathway_hypergraph import (
    module_top_genes,
    pathway_activation,
)

MEMBERSHIP = {
    "P1": {"100", "101", "102", "103", "104"},
    "P2": {"200", "201"},
}


def test_pathway_activation_sums_scores() -> None:
    activation = pathway_activation(MEMBERSHIP, ["100", "101", "999"], [2.0, 3.0, 9.0])
    assert activation["P1"] == 5.0
    assert "P2" not in activation


def test_module_top_genes() -> None:
    top = module_top_genes(
        MEMBERSHIP, "P1", ["104", "100", "101"], [9.0, 1.0, 2.0], max_genes=2
    )
    assert top == ["104", "101"]


def test_module_loss_prefers_supported_spread_modules() -> None:
    supported = {g: 0.1 for g in ("100", "101", "102", "103", "104")}
    hub = {g: (0.9 if g == "100" else 0.01) for g in ("100", "101", "102", "103", "104")}
    sparse = {"100": 0.1, "101": 0.1}  # unsupported (< 5 genes)
    loss_supported, metrics_supported = module_regularization_loss(
        {"P1": 1.0}, MEMBERSHIP, gene_scores_by_gene=supported
    )
    loss_hub, _ = module_regularization_loss(
        {"P1": 1.0}, MEMBERSHIP, gene_scores_by_gene=hub
    )
    loss_sparse, metrics_sparse = module_regularization_loss(
        {"P1": 1.0}, MEMBERSHIP, gene_scores_by_gene=sparse
    )
    assert loss_hub >= loss_supported
    assert metrics_sparse["unsupported_count"] >= 1
