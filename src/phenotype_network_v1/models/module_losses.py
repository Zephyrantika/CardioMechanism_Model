"""Module regularisation losses for V1 (Milestone 5).

Penalises pathway modules that would be rejected downstream: collapsed
(too few supported genes), single-hub-dominated (one gene carries the
activation), or unsupported (no frozen Reactome/GO evidence). The loss is a
soft penalty; hard rejection happens in the module audit.
"""

from __future__ import annotations

import math
from typing import Any


def _entropy(weights: list[float]) -> float:
    total = sum(weights)
    if total <= 0:
        return 0.0
    normalised = [w / total for w in weights]
    return -sum(w * math.log(w) for w in normalised if w > 0)


def module_regularization_loss(
    pathway_activation: dict[str, float],
    membership: dict[str, set[str]],
    *,
    gene_scores_by_gene: dict[str, float],
    min_module_genes: int = 5,
) -> tuple[float, dict[str, Any]]:
    """Soft regulariser over a query's pathway activations.

    For each activated pathway, the module score is ``log1p(activation)``
    gated by how many candidate genes inside the pathway are supported, with a
    hub penalty when a single gene dominates the pathway's gene-level scores.
    """
    loss = 0.0
    metrics: dict[str, Any] = {
        "activated_pathways": len(pathway_activation),
        "unsupported_count": 0,
        "hub_dominated_count": 0,
        "collapsed_count": 0,
    }
    for pathway_id, activation in pathway_activation.items():
        pathway_genes = membership.get(pathway_id, set())
        scores = [
            gene_scores_by_gene[gene_id]
            for gene_id in pathway_genes
            if gene_id in gene_scores_by_gene and gene_scores_by_gene[gene_id] > 0
        ]
        supported = len(scores)
        if supported < min_module_genes:
            metrics["unsupported_count"] += 1
            # Mild penalty: encourage supported modules.
            loss += 1.0 / max(1, min_module_genes - supported + 1)
            continue
        hub_penalty = 1.0 - _entropy(scores) / max(1e-8, math.log(supported))
        if hub_penalty > 0.5:
            metrics["hub_dominated_count"] += 1
        loss += hub_penalty * 0.1
    loss = loss / max(1, metrics["activated_pathways"])
    return float(loss), metrics


__all__ = ["module_regularization_loss"]
