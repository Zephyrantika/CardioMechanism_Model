"""Phenotype-conditioned basis-decomposed R-GCN (V1 Milestone 3).

Two-layer basis R-GCN whose relation messages are FiLM-conditioned on the
query vector ``q_d`` each layer, with LayerNorm residual updates and
recurrent phenotype-seed injection at every layer:

    m_v^(l)   = sum_r sum_{u in N_r(v)} norm(u,v,r) * W_r^(l) h_u^(l)
    h_v^(l+1) = LayerNorm(h_v^(l) + act(gamma_l(q_d) * m_v^(l) + beta_l(q_d)))
    h_seed   ^= rho * seed_state + (1 - rho) * h_seed

``seed_state`` is the initial HPO-seed node representation, which re-anchors
the phenotype constraint at every layer (recurrent injection). Disabling the
query conditioning (constant FiLM), the seed injection (rho=0), or using the
query only in a final decoder are explicit ablations.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torch.nn import functional as F


def _relation_messages(
    x: torch.Tensor,
    edge_index: torch.Tensor,
    edge_type: torch.Tensor,
    degree_norm: torch.Tensor,
    relation_weights: list[torch.Tensor],
) -> torch.Tensor:
    """Sum relation-specific linear messages with symmetric normalisation."""
    row, col = edge_index
    source = x[row]
    out = torch.zeros_like(x)
    for rel, weight in enumerate(relation_weights):
        mask = edge_type == rel
        if not bool(mask.any()):
            continue
        src = source[mask]
        messages = src @ weight
        messages = messages * degree_norm[col[mask]].unsqueeze(-1)
        out.index_add_(0, col[mask], messages)
    return out


class ConditionedRGCN(nn.Module):
    """Basis-decomposed R-GCN with per-layer FiLM conditioning."""

    def __init__(
        self,
        num_nodes: int,
        num_relations: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        relation_bases: int = 8,
        *,
        rho: float = 0.20,
        dropout: float = 0.20,
        seed: int = 42,
    ) -> None:
        super().__init__()
        torch.manual_seed(seed)
        self.num_layers = num_layers
        self.rho = float(rho)
        self.node_embedding = nn.Embedding(num_nodes, hidden_dim)
        nn.init.normal_(self.node_embedding.weight, std=0.02)
        # Basis decomposition: W_r = sum_b a_{r,b} B_b
        self.bases = nn.ParameterList(
            [
                nn.Parameter(torch.empty(hidden_dim, hidden_dim))
                for _ in range(relation_bases)
            ]
        )
        for base in self.bases:
            nn.init.xavier_uniform_(base)
        self.basis_coefficients = nn.Parameter(
            torch.randn(num_relations, relation_bases) * 0.1
        )
        self.film_gamma = nn.ModuleList(
            [nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers)]
        )
        self.film_beta = nn.ModuleList(
            [nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers)]
        )
        self.layer_norms = nn.ModuleList(
            [nn.LayerNorm(hidden_dim) for _ in range(num_layers)]
        )
        self.dropout = nn.Dropout(dropout)

    def _relation_weight(self, layer: int) -> torch.Tensor:
        # Basis coefficients are shared across layers in this implementation.
        coefficients = F.softmax(self.basis_coefficients, dim=-1)  # [R, B]
        stacked = torch.stack([base for base in self.bases], dim=0)  # [B, H, H]
        return torch.einsum("rb,bhi->rhi", coefficients, stacked)

    def forward(
        self,
        graph: dict[str, torch.Tensor],
        q_d: torch.Tensor,
        seed_hpo_local: list[int],
        hpo_offset: int,
        *,
        condition: bool = True,
        seed_injection: bool = True,
    ) -> torch.Tensor:
        """Return hidden states [num_nodes, hidden] for one query."""
        x = self.node_embedding.weight
        if not condition:
            q_d = torch.zeros_like(q_d)
        seed_state = None
        if seed_injection and seed_hpo_local:
            seed_global = torch.tensor(
                [hpo_offset + index for index in seed_hpo_local], dtype=torch.long
            )
            seed_state = x[seed_global].clone()

        row, col = graph["edge_index"]
        degree = torch.zeros(int(x.size(0)), device=x.device)
        degree.index_add_(0, col, torch.ones_like(col, dtype=x.dtype))
        degree = degree.clamp(min=1.0)
        degree_norm = 1.0 / degree.sqrt()

        for layer in range(self.num_layers):
            gamma = torch.sigmoid(self.film_gamma[layer](q_d)).unsqueeze(0)
            beta = self.film_beta[layer](q_d).unsqueeze(0)
            messages = _relation_messages(
                x, graph["edge_index"], graph["edge_type"],
                degree_norm, self._relation_weight(layer).unbind(0),
            )
            conditioned = gamma * messages + beta
            x = x + F.gelu(conditioned)
            x = self.layer_norms[layer](x)
            if seed_injection and seed_state is not None:
                x[seed_global] = (
                    self.rho * seed_state + (1.0 - self.rho) * x[seed_global]
                )
            x = self.dropout(x)
        return x


__all__ = ["ConditionedRGCN"]
