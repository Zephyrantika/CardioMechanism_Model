"""Graph convolutional network baseline for V1 (Milestone 2).

Homogeneous GCN comparator over the unified V1 graph. Shared helpers for
turning a PyG HeteroData (M1 artifact) into a unified node/edge indexing are
defined here and reused by ``rgcn.py`` and ``hgt.py``.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torch_geometric.nn import GCNConv

NODE_TYPES = ("phenotype", "gene", "pathway")


def build_unified_from_stores(heterodata: Any) -> dict[str, torch.Tensor]:
    """Build the unified representation by iterating edge stores directly."""
    offsets: dict[str, int] = {}
    sizes: dict[str, int] = {}
    running = 0
    for node_type in NODE_TYPES:
        sizes[node_type] = int(heterodata[node_type].num_nodes)
        offsets[node_type] = running
        running += sizes[node_type]

    rows: list[int] = []
    cols: list[int] = []
    types: list[int] = []
    relation_names: list[str] = []
    for (src_type, relation, dst_type) in heterodata.edge_types:
        edge_index = heterodata[src_type, relation, dst_type].edge_index
        rel_index = len(relation_names)
        relation_names.append(relation)
        rows.extend((edge_index[0] + offsets[src_type]).tolist())
        cols.extend((edge_index[1] + offsets[dst_type]).tolist())
        types.extend([rel_index] * edge_index.size(1))
    return {
        "num_nodes": int(running),
        "sizes": sizes,
        "offsets": offsets,
        "edge_index": torch.tensor([rows, cols], dtype=torch.long),
        "edge_type": torch.tensor(types, dtype=torch.long),
        "num_relations": len(relation_names),
        "relation_names": relation_names,
    }


class GCNBaseline(nn.Module):
    """Homogeneous GCN gene-ranking baseline with seed injection."""

    def __init__(
        self,
        num_nodes: int,
        gene_node_indices: list[int],
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
        layers = []
        for _ in range(num_layers):
            layers.append(GCNConv(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
        self.convs = nn.ModuleList(layers)
        self.head = nn.Linear(hidden_dim, 1)

    def forward(
        self,
        graph: dict[str, torch.Tensor],
        seed_hpo_local: list[int],
        hpo_offset: int,
    ) -> torch.Tensor:
        """Return logits over gene nodes for one query (seed HPO indices)."""
        x = self.node_embedding.weight.clone()
        if seed_hpo_local:
            seed_global = torch.tensor(
                [hpo_offset + index for index in seed_hpo_local], dtype=torch.long
            )
            x[seed_global] = x[seed_global] + self.seed_vector
        edge_index = graph["edge_index"]
        for layer in self.convs:
            if isinstance(layer, GCNConv):
                x = layer(x, edge_index)
            else:
                x = layer(x)
        gene_x = x[self.gene_node_indices]
        return self.head(gene_x).squeeze(-1)


__all__ = [
    "GCNBaseline",
    "NODE_TYPES",
    "build_unified_from_stores",
]
