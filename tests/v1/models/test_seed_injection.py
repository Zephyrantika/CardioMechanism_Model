"""Recurrent phenotype-seed injection tests (Milestone 3)."""

from __future__ import annotations

import torch

from _m3_helpers import gene_node_indices, hpo_offset, small_graph

from phenotype_network_v1.models.conditioned_rgcn import ConditionedRGCN


def _hidden(seed: int, seed_injection: bool, rho: float, hpo: list[int]) -> torch.Tensor:
    torch.manual_seed(seed)
    backbone = ConditionedRGCN(
        9, 3, hidden_dim=16, num_layers=2, relation_bases=2, rho=rho, seed=seed
    )
    backbone.eval()
    with torch.no_grad():
        q = torch.randn(16)
        return backbone(small_graph(), q, hpo, hpo_offset(), seed_injection=seed_injection)


def test_seed_injection_changes_output() -> None:
    with_seed = _hidden(seed=1, seed_injection=True, rho=0.2, hpo=[2])
    without = _hidden(seed=1, seed_injection=False, rho=0.2, hpo=[2])
    genes = torch.tensor(gene_node_indices(), dtype=torch.long)
    assert not torch.allclose(with_seed[genes], without[genes])


def test_zero_rho_matches_no_injection() -> None:
    """rho=0 must reproduce the no-seed-injection behaviour."""
    zero_rho = _hidden(seed=2, seed_injection=True, rho=0.0, hpo=[0])
    off = _hidden(seed=2, seed_injection=False, rho=0.0, hpo=[0])
    genes = torch.tensor(gene_node_indices(), dtype=torch.long)
    assert torch.allclose(zero_rho[genes], off[genes], atol=1e-6)


def test_seed_without_local_hpo_is_safe() -> None:
    torch.manual_seed(3)
    backbone = ConditionedRGCN(
        9, 3, hidden_dim=16, num_layers=2, relation_bases=2, rho=0.2, seed=3
    )
    backbone.eval()
    with torch.no_grad():
        q = torch.zeros(16)
        out = backbone(small_graph(), q, [], hpo_offset())
    assert torch.isfinite(out).all()
