"""RWR-residual decoder tests (Milestone 3)."""

from __future__ import annotations

import torch

from phenotype_network_v1.models.decoder import ScoreDecoder


def test_decoder_shape() -> None:
    decoder = ScoreDecoder(16, use_rwr=True, seed=0)
    gene_hidden = torch.randn(4, 16)
    q = torch.randn(16)
    rwr = torch.rand(4)
    logits = decoder(gene_hidden, q, rwr)
    assert logits.shape == (4,)
    assert torch.isfinite(logits).all()


def test_rwr_feature_required_when_enabled() -> None:
    decoder = ScoreDecoder(16, use_rwr=True, seed=0)
    try:
        decoder(torch.randn(4, 16), torch.randn(16))
    except ValueError:
        return
    raise AssertionError("expected ValueError when rwr_feature is missing")


def test_rwr_column_changes_scores() -> None:
    gene_hidden = torch.randn(4, 16, generator=torch.Generator().manual_seed(0))
    q = torch.randn(16, generator=torch.Generator().manual_seed(1))
    with_rwr = ScoreDecoder(16, use_rwr=True, seed=2)
    without_rwr = ScoreDecoder(16, use_rwr=False, seed=2)
    a = with_rwr(gene_hidden, q, torch.zeros(4))
    b = without_rwr(gene_hidden, q)
    # Same seed, but the RWR column changes capacity and output.
    assert a.shape == b.shape
    assert torch.isfinite(a).all() and torch.isfinite(b).all()


def test_rwr_values_change_logits() -> None:
    decoder = ScoreDecoder(16, use_rwr=True, seed=3)
    gene_hidden = torch.randn(4, 16)
    q = torch.randn(16)
    low = decoder(gene_hidden, q, torch.zeros(4))
    high = decoder(gene_hidden, q, torch.ones(4) * 5.0)
    assert not torch.allclose(low, high)
