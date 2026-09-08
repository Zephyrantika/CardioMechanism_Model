"""Gene score decoder with RWR residual (V1 Milestone 3).

    score(d, g) = MLP([h_g, q_d, h_g * q_d, log1p(rwr_score(d, g))])

``rwr_score`` are frozen V0 values passed in per gene; the optional RWR column
is dropped for the mandatory ``no_rwr_residual`` ablation. Conditioning only
in the decoder is the ``condition_decoder_only`` ablation.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class ScoreDecoder(nn.Module):
    """Per-gene score head combining node state, query, and RWR residual."""

    def __init__(
        self,
        hidden_dim: int,
        *,
        use_rwr: bool = True,
        seed: int = 42,
    ) -> None:
        super().__init__()
        torch.manual_seed(seed)
        self.use_rwr = use_rwr
        in_features = 2 * hidden_dim + hidden_dim  # h_g, q, h_g*q
        if use_rwr:
            in_features += 1
        self.mlp = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        gene_hidden: torch.Tensor,  # [n_gene, hidden]
        q_d: torch.Tensor,           # [hidden]
        rwr_feature: torch.Tensor | None = None,  # [n_gene] frozen log1p RWR
    ) -> torch.Tensor:
        """Return per-gene logits [n_gene] for one query."""
        h = gene_hidden
        q = q_d.unsqueeze(0).expand_as(h)
        features = [h, q, h * q]
        if self.use_rwr:
            if rwr_feature is None:
                raise ValueError("use_rwr=True requires an rwr_feature tensor")
            features.append(rwr_feature.unsqueeze(-1).to(h.dtype))
        return self.mlp(torch.cat(features, dim=-1)).squeeze(-1)


__all__ = ["ScoreDecoder"]
