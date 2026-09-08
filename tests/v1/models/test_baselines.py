"""CPU synthetic tests for V1 deep baselines (Milestone 2).

Each baseline must run on a small graph without CUDA, produce one score per
gene, and be able to overfit a tiny training signal.
"""

from __future__ import annotations

import torch

from phenotype_network_v1.models.gcn import GCNBaseline
from phenotype_network_v1.models.hgt import HGTBaseline
from phenotype_network_v1.models.rgcn import RGCNBaseline

NUM_PHENOTYPE = 5
NUM_GENE = 4


def _small_graph(seed: int = 0) -> dict:
    torch.manual_seed(seed)
    # phenotype nodes 0..4, gene nodes 5..8; a few typed edges.
    edge_index = torch.tensor(
        [
            [0, 1, 2, 5, 6, 7],
            [1, 2, 5, 6, 7, 8],
        ],
        dtype=torch.long,
    )
    edge_type = torch.tensor([0, 0, 1, 1, 2, 2], dtype=torch.long)
    return {
        "num_nodes": NUM_PHENOTYPE + NUM_GENE,
        "edge_index": edge_index,
        "edge_type": edge_type,
        "num_relations": 3,
        "relation_names": ["hpo_hpo", "hpo_gene", "gene_gene"],
    }


def _make_model(name: str) -> torch.nn.Module:
    gene_indices = list(range(NUM_PHENOTYPE, NUM_PHENOTYPE + NUM_GENE))
    common = {
        "num_nodes": NUM_PHENOTYPE + NUM_GENE,
        "gene_node_indices": gene_indices,
        "hidden_dim": 16,
        "num_layers": 2,
        "seed": 42,
    }
    if name == "gcn":
        return GCNBaseline(**common)
    if name == "rgcn":
        return RGCNBaseline(num_relations=3, **common)
    return HGTBaseline(num_relations=3, heads=2, edge_dim=4, **common)


def _score(model: torch.nn.Module, seed_hpo: list[int]) -> torch.Tensor:
    model.eval()
    with torch.no_grad():
        return model(_small_graph(), seed_hpo, hpo_offset=0)


def test_each_baseline_scores_all_genes() -> None:
    for name in ("gcn", "rgcn", "hgt"):
        model = _make_model(name)
        scores = _score(model, [2])
        assert scores.shape == (NUM_GENE,)
        assert torch.isfinite(scores).all()


def test_baseline_deterministic_replay() -> None:
    for name in ("gcn", "rgcn", "hgt"):
        model = _make_model(name)
        first = _score(model, [2])
        second = _score(model, [2])
        assert torch.allclose(first, second), name


def test_distinct_queries_produce_distinct_scores() -> None:
    for name in ("gcn", "rgcn", "hgt"):
        model = _make_model(name)
        empty_seed = _score(model, [])
        seeded = _score(model, [2])
        assert not torch.allclose(empty_seed, seeded), name


def test_baseline_overfits_tiny_signal() -> None:
    """Each baseline can lower BCE on a single synthetic query."""
    graph = _small_graph()
    for name in ("gcn", "rgcn", "hgt"):
        torch.manual_seed(0)
        model = _make_model(name)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.05)
        loss_fn = torch.nn.BCEWithLogitsLoss()
        targets = torch.zeros(NUM_GENE)
        targets[0] = 1.0  # gene local 0 is a 1-hop neighbour of HPO node 2
        model.train()
        initial = None
        for _ in range(60):
            optimizer.zero_grad()
            logits = model(graph, [2], hpo_offset=0)
            loss = loss_fn(logits, targets)
            if initial is None:
                initial = float(loss.detach())
            loss.backward()
            optimizer.step()
        final = float(loss.detach())
        assert final < initial, f"{name}: final {final} not below initial {initial}"
