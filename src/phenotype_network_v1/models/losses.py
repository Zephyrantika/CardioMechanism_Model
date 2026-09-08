"""Loss utilities for V1 conditioned training (Milestone 3).

Milestone 3 uses binary classification on covered positive genes with optional
per-sample weights. Milestone 4 adds PU bagging losses on top; the unlabelled
policy (never confirmed negative) is enforced at the data level, not here.
"""

from __future__ import annotations

import torch
from torch.nn import functional as F


def classification_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    sample_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    """Binary cross-entropy with logits and optional per-gene weights."""
    if logits.shape != targets.shape:
        raise ValueError("logits and targets must share shape")
    loss = F.binary_cross_entropy_with_logits(
        logits, targets, weight=sample_weight, reduction="mean"
    )
    return loss


def positive_recall_at_k(scores: torch.Tensor, targets: torch.Tensor, k: int = 10) -> float:
    """Fraction of positive genes found in the top-k ranked scores."""
    if k <= 0:
        raise ValueError("k must be positive")
    if targets.sum() == 0:
        return float("nan")
    topk = torch.topk(scores, k=min(k, scores.size(0))).indices
    return float(targets[topk].sum() / targets.sum())


__all__ = ["classification_loss", "positive_recall_at_k"]
