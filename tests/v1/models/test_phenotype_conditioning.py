"""Conditioning tests: distinct queries differ; replay is deterministic."""

from __future__ import annotations

import torch

from _m3_helpers import gene_node_indices, hpo_offset, small_graph

from phenotype_network_v1.models.conditioned_rgcn import ConditionedRGCN
from phenotype_network_v1.models.decoder import ScoreDecoder
from phenotype_network_v1.models.phenotype_encoder import QueryPhenotypeEncoder


def _components(attention: bool = True, rho: float = 0.2, seed: int = 0):
    torch.manual_seed(seed)
    encoder = QueryPhenotypeEncoder(5, 16, attention=attention, seed=seed)
    backbone = ConditionedRGCN(
        9, 3, hidden_dim=16, num_layers=2, relation_bases=2, rho=rho, seed=seed
    )
    decoder = ScoreDecoder(16, use_rwr=False, seed=seed)
    return encoder, backbone, decoder


def _score(encoder, backbone, decoder, hpo: list[int], w: list[float]) -> torch.Tensor:
    backbone.eval(); encoder.eval(); decoder.eval()
    with torch.no_grad():
        q, _ = encoder(torch.tensor(hpo, dtype=torch.long), torch.tensor(w))
        hidden = backbone(small_graph(), q, hpo, hpo_offset())
        gene_hidden = hidden[torch.tensor(gene_node_indices(), dtype=torch.long)]
        return decoder(gene_hidden, q)


def test_distinct_queries_produce_distinct_gene_scores() -> None:
    encoder, backbone, decoder = _components()
    first = _score(encoder, backbone, decoder, [0, 1], [1.0, 1.0])
    second = _score(encoder, backbone, decoder, [2, 3], [1.0, 1.0])
    assert first.shape == (4,)
    assert not torch.allclose(first, second)


def test_deterministic_replay() -> None:
    encoder, backbone, decoder = _components()
    first = _score(encoder, backbone, decoder, [0, 2], [2.0, 0.5])
    second = _score(encoder, backbone, decoder, [0, 2], [2.0, 0.5])
    assert torch.allclose(first, second)


def test_weighted_mean_pool_ablation_runs() -> None:
    encoder, backbone, decoder = _components(attention=False)
    scores = _score(encoder, backbone, decoder, [0, 1], [1.0, 3.0])
    assert torch.isfinite(scores).all()


def test_no_query_conditioning_gives_identical_scores() -> None:
    """Constant FiLM means the query cannot change backbone output."""
    encoder, backbone, decoder = _components()
    backbone.eval(); encoder.eval(); decoder.eval()
    with torch.no_grad():
        q_a = torch.randn(16)
        q_b = torch.randn(16)
        h_a = backbone(small_graph(), q_a, [2], hpo_offset(), condition=False)
        h_b = backbone(small_graph(), q_b, [2], hpo_offset(), condition=False)
        g = torch.tensor(gene_node_indices(), dtype=torch.long)
        assert torch.allclose(h_a[g], h_b[g])
        # And the query does change outputs when conditioning is enabled.
        h_cond = backbone(small_graph(), q_a, [2], hpo_offset(), condition=True)
        assert not torch.allclose(h_cond[g], h_b[g])
