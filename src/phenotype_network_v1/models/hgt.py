"""HGT-style relation-conditioned attention baseline for V1 (Milestone 2).

Parameter-matched comparator: multi-head, relation-conditioned transformer
message passing over the unified V1 graph. Relation identities condition the
attention through edge attributes, following the HGT idea of relation-aware
attention, implemented on top of PyG's TransformerConv. See
``docs/v1_literature_reproducibility.md`` for the exact deviation notes.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torch_geometric.nn import TransformerConv


class HGTBaseline(nn.Module):
    """Relation-conditioned multi-head attention gene-ranking baseline."""

    def __init__(
        self,
        num_nodes: int,
        gene_node_indices: list[int],
        num_relations: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        heads: int = 4,
        edge_dim: int = 16,
        dropout: float = 0.1,
        seed: int = 42,
    ) -> None:
        super().__init__()
        torch.manual_seed(seed)
        if hidden_dim % heads != 0:
            raise ValueError("hidden_dim must be divisible by heads")
        self.gene_node_indices = torch.tensor(gene_node_indices, dtype=torch.long)
        self.num_relations = num_relations
        self.node_embedding = nn.Embedding(num_nodes, hidden_dim)
        self.seed_vector = nn.Parameter(torch.zeros(hidden_dim))
        nn.init.normal_(self.seed_vector, std=0.02)
        self.relation_embedding = nn.Embedding(num_relations, edge_dim)
        self.layers = nn.ModuleList(
            [
                TransformerConv(
                    hidden_dim,
                    hidden_dim // heads,
                    heads=heads,
                    edge_dim=edge_dim,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )
        self.head = nn.Linear(hidden_dim, 1)

    def forward(
        self,
        graph: dict[str, torch.Tensor],
        seed_hpo_local: list[int],
        hpo_offset: int,
    ) -> torch.Tensor:
        x = self.node_embedding.weight.clone()
        if seed_hpo_local:
            seed_global = torch.tensor(
                [hpo_offset + index for index in seed_hpo_local], dtype=torch.long
            )
            x[seed_global] = x[seed_global] + self.seed_vector
        edge_index = graph["edge_index"]
        edge_type = graph["edge_type"]
        edge_attr = self.relation_embedding(edge_type)
        num_nodes = int(x.size(0))
        for layer in self.layers:
            x = torch.relu(layer(x, edge_index, edge_attr=edge_attr))
        gene_x = x[self.gene_node_indices]
        return self.head(gene_x).squeeze(-1)


__all__ = ["HGTBaseline"]
