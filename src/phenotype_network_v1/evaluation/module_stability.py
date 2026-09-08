"""Soft mechanism-module stability and audit for V1 (Milestone 5).

Extracts per-query pathway modules from ranked gene scores and rejects
collapsed, single-hub-dominated, unstable, or unsupported modules into an
audit table. Module membership labels come from frozen Reactome/GO evidence
only. The V0 Leiden comparator is retained unchanged and is not replaced.
"""

from __future__ import annotations

from typing import Any


def jaccard(first: set[str], second: set[str]) -> float:
    union = first | second
    if not union:
        return 1.0
    return len(first & second) / len(union)


def audit_module(
    module_genes: set[str],
    *,
    gene_scores: dict[str, float],
    min_size: int = 5,
    max_size: int = 100,
    hub_fraction: float = 0.8,
    min_stability: float = 0.0,
    stability: float = 1.0,
) -> list[str]:
    """Return rejection reasons for one module (empty list = retained)."""
    reasons: list[str] = []
    if len(module_genes) < min_size:
        reasons.append("collapsed")
    if len(module_genes) > max_size:
        reasons.append("oversized")
    if module_genes:
        scores = sorted(
            (gene_scores.get(gene_id, 0.0) for gene_id in module_genes),
            reverse=True,
        )
        top_fraction = scores[0] / max(1e-9, sum(scores))
        if top_fraction >= hub_fraction:
            reasons.append("single_hub_dominated")
    if stability < min_stability:
        reasons.append("unstable")
    if not module_genes:
        reasons.append("unsupported")
    return reasons


def extract_modules(
    pathway_activation: dict[str, float],
    *,
    membership: dict[str, set[str]],
    gene_ids: list[str],
    gene_scores: list[float],
    max_module_genes: int = 100,
) -> dict[str, set[str]]:
    """Top supported genes per activated pathway (soft modules)."""
    score_by_gene = dict(zip(gene_ids, gene_scores))
    modules: dict[str, set[str]] = {}
    for pathway_id in pathway_activation:
        pathway_genes = membership.get(pathway_id, set())
        supported = {
            gene_id
            for gene_id in pathway_genes
            if gene_id in score_by_gene and score_by_gene[gene_id] > 0
        }
        if supported:
            modules[pathway_id] = set(
                sorted(supported, key=lambda g: score_by_gene[g], reverse=True)[
                    :max_module_genes
                ]
            )
    return modules


def split_half_stability(
    modules: dict[str, set[str]],
    gene_ids_a: list[str],
    scores_a: list[float],
    gene_ids_b: list[str],
    scores_b: list[float],
) -> dict[str, float]:
    """Per-pathway stability between two independent ranking halves."""
    def _supported(module: set[str], gene_ids: list[str], scores: list[float]) -> set[str]:
        present = set(gene_ids)
        score_map = dict(zip(gene_ids, scores))
        return {
            gene_id
            for gene_id in module
            if gene_id in present and score_map[gene_id] > 0
        }

    stability: dict[str, float] = {}
    ids_a, ids_b = set(gene_ids_a), set(gene_ids_b)
    for pathway_id, module in modules.items():
        support_a = {g for g in module if g in ids_a}
        support_b = {g for g in module if g in ids_b}
        stability[pathway_id] = jaccard(support_a, support_b)
    return stability


__all__ = [
    "audit_module",
    "extract_modules",
    "jaccard",
    "split_half_stability",
]
