"""Counterfactual fidelity evaluation for V1 path explanations (M6).

For each evidence path to a gene, remove one critical edge and measure the
drop in the gene's score; larger drops mean the path is more faithful. The
evaluation never uses held-out labels to search or score paths.
"""

from __future__ import annotations

from typing import Any, Callable


def counterfactual_fidelity(
    *,
    path_nodes: list[int],
    path_relations: list[str],
    score_gene: Callable[[], float],
    score_gene_without_edge: Callable[[int, int], float],
) -> dict[str, Any]:
    """Fidelity of one path: baseline score vs edge-deletion scores.

    ``score_gene`` returns the gene score on the intact graph;
    ``score_gene_without_edge(u, v)`` returns the gene score after removing
    edge (u, v). All edges of the path are counterfactually removed one at a
    time; the minimum delta is reported as the path's fidelity.
    """
    baseline = float(score_gene())
    deletions = []
    deltas = []
    for index in range(len(path_nodes) - 1):
        source = path_nodes[index]
        target = path_nodes[index + 1]
        deletion = float(score_gene_without_edge(source, target))
        deletions.append({"edge": [source, target], "score_without": deletion})
        deltas.append(baseline - deletion)
    return {
        "baseline_score": baseline,
        "edge_deletions": deletions,
        "deltas": deltas,
        "fidelity": float(min(deltas)) if deltas else 0.0,
        "mean_fidelity": float(sum(deltas) / len(deltas)) if deltas else 0.0,
    }


__all__ = ["counterfactual_fidelity"]
