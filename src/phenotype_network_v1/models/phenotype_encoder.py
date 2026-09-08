"""Query phenotype encoder for V1 (Milestone 3).

For disease ``d`` builds the query vector by attention-pooling weighted HPO
embeddings:

    q_d = AttentionPool({w_dh * e_h : h in HPO(d)})

Absent HPO terms are masked by construction (only present indices and their
positive weights are passed in) and normalised attention weights are exposed
for audit. A weighted mean is available as the required ablation.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class QueryPhenotypeEncoder(nn.Module):
    """Attention-pooled weighted HPO query encoder."""

    def __init__(
        self,
        num_hpo: int,
        hidden_dim: int,
        *,
        attention: bool = True,
        seed: int = 42,
    ) -> None:
        super().__init__()
        torch.manual_seed(seed)
        self.num_hpo = num_hpo
        self.hidden_dim = hidden_dim
        self.attention = attention
        self.hpo_embeddings = nn.Embedding(num_hpo, hidden_dim)
        nn.init.normal_(self.hpo_embeddings.weight, std=0.02)
        if attention:
            self.attention_projection = nn.Linear(hidden_dim, 1)

    def forward(
        self, hpo_indices: torch.Tensor, hpo_weights: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (q_d, attention_weights) for one disease.

        ``hpo_indices`` are local HPO node indices (long), ``hpo_weights`` the
        matching frequency*IC weights (positive, >= 0). Absent HPO terms are
        simply not present, so they cannot contribute.
        """
        embeddings = self.hpo_embeddings(hpo_indices)  # [k, hidden]
        weights = hpo_weights.clamp(min=0.0)
        if embeddings.size(0) == 0:
            return torch.zeros(self.hidden_dim, device=embeddings.device), torch.zeros(
                0, device=embeddings.device
            )
        if not self.attention:
            # Weighted-mean ablation (non-attention pool).
            total = weights.sum().clamp(min=1e-8)
            q = (embeddings * weights.unsqueeze(-1)).sum(dim=0) / total
            attention_weights = weights / total
            return q, attention_weights
        logits = self.attention_projection(embeddings).squeeze(-1)
        # Frequency weights gate attention; log(0) -> -inf masks effectively.
        logits = logits + torch.log(weights.clamp(min=1e-12))
        attention_weights = F.softmax(logits, dim=0)
        q = (embeddings * attention_weights.unsqueeze(-1)).sum(dim=0)
        return q, attention_weights


__all__ = ["QueryPhenotypeEncoder"]
