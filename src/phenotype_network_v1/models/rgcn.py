"""Relation-aware graph convolutional baseline for V1 (Milestone 2).

Unconditional R-GCN comparator over the unified V1 graph: one linear
transformation per relation, summed with a self loop, mirroring PyG's
RGCNConv semantics on the relation-typed unified edge index.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torch.nn import functional as F


class _RGCNLayer(nn.Module):
    """Relation-specific message transform with self-loop residual."""

    def __init__(self, hidden_dim: int, num_relations: int) -> None:
        super().__init__()
        self.relation_weights = nn.ParameterList(
            [nn.Parameter(torch.empty(hidden_dim, hidden_dim)) for _ in range(num_relations)]
        )
        self.loop = nn.Linear(hidden_dim, hidden_dim)
        for parameter in self.relation_weights:
            nn.init.xavier_uniform_(parameter)
        nn.init.xavier_uniform_(self.loop.weight)

    def forward(
        self, x: torch.Tensor, edge_index: torch.Tensor, edge_type: torch.Tensor
    ) -> torch.Tensor:
        out = self.loop(x)
        row, col = edge_index
        source = x[row]
        for rel in range(len(self.relation_weights)):
            mask = edge_type == rel
            if not bool(mask.any()):
                continue
            source_rel = source[mask]
            messages = source_rel @ self.relation_weights[rel]
            out.index_add_(0, col[mask], messages)
        return out


class RGCNBaseline(nn.Module):
    """Unconditional relation-aware GCN gene-ranking baseline."""

    def __init__(
        self,
        num_nodes: int,
        gene_node_indices: list[int],
        num_relations: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1,
        seed: int = 42,
    ) -> None:
        super().__init__()
        torch.manual_seed(seed)
        self.gene_node_indices = torch.tensor(gene_node_indices, dtype=torch.long)
        self.hidden_dim = hidden_dim
        self.node_embedding = nn.Embedding(num_nodes, hidden_dim)
        self.seed_vector = nn.Parameter(torch.zeros(hidden_dim))
        nn.init.normal_(self.seed_vector, std=0.02)
        self.layers = nn.ModuleList(
            [_RGCNLayer(hidden_dim, num_relations) for _ in range(num_layers)]
        )
        self.dropout = nn.Dropout(dropout)
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
        for layer in self.layers:
            x = F.relu(layer(x, edge_index, edge_type))
            x = self.dropout(x)
        gene_x = x[self.gene_node_indices]
        return self.head(gene_x).squeeze(-1)


__all__ = ["RGCNBaseline"]
