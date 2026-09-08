"""Relation-aware path reasoning for V1 (Milestone 6).

Bounded, structure-only path search from a disease's HPO seed nodes to its
candidate genes over the frozen fold graph. Only approved biomedical
relations are traversed and paths respect a maximum length. Held-out labels
are never used to search or score paths; unresolved targets are preserved and
reported rather than silently dropped.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# Approved biomedical relations (V0 graph relation names).
APPROVED_RELATIONS = {
    "hpo_hpo",
    "hpo_gene",
    "gene_hpo",
    "gene_gene",
    "gene_pathway",
    "pathway_gene",
    "pathway_pathway",
}

MAX_DEFAULT_LENGTH = 4


@dataclass
class PathResult:
    """One candidate evidence path (global V0 node indices + relations)."""

    nodes: list[int]
    relations: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {"nodes": self.nodes, "relations": self.relations}


def build_adjacency(
    edge_index: Any,
    edge_type: Any,
    relation_names: list[str],
    *,
    approved: set[str] | None = None,
) -> dict[int, list[tuple[int, str]]]:
    """Forward adjacency {src_global: [(dst_global, relation), ...]}."""
    approved = APPROVED_RELATIONS if approved is None else approved
    adjacency: dict[int, list[tuple[int, str]]] = {}
    rows = edge_index[0].tolist()
    cols = edge_index[1].tolist()
    types = edge_type.tolist()
    for src, dst, rel_index in zip(rows, cols, types):
        relation = relation_names[rel_index]
        if relation in approved:
            adjacency.setdefault(src, []).append((dst, relation))
    return adjacency


def reason_paths(
    adjacency: dict[int, list[tuple[int, str]]],
    seed_nodes: list[int],
    target_nodes: set[int],
    *,
    max_length: int = MAX_DEFAULT_LENGTH,
    max_paths: int = 20,
    max_branches: int = 8,
) -> list[PathResult]:
    """Bounded BFS search for gene-target paths from the HPO seed nodes.

    Paths are length-limited, cycle-free, and terminate at the first target
    node reached (unresolved targets simply are not in ``target_nodes`` and
    are reported separately by the caller). Deterministic ordering.
    """
    if max_length <= 0:
        raise ValueError("max_length must be positive")
    results: list[PathResult] = []
    queue: list[list[int]] = [[seed] for seed in seed_nodes]
    while queue and len(results) < max_paths:
        path = queue.pop(0)
        last = path[-1]
        if len(path) > 1 and last in target_nodes:
            results.append(_to_result(path, adjacency))
            continue
        if len(path) >= max_length + 1:
            continue
        neighbours = adjacency.get(last, [])[:max_branches]
        for neighbour, relation in neighbours:
            if neighbour in path:
                continue
            queue.append(path + [neighbour])
    return results


def _to_result(path: list[int], adjacency: dict[int, list[tuple[int, str]]]) -> PathResult:
    relations: list[str] = []
    for index in range(len(path) - 1):
        candidates = dict((dst, rel) for dst, rel in adjacency.get(path[index], []))
        relations.append(candidates.get(path[index + 1], "unknown"))
    return PathResult(nodes=path, relations=relations)


__all__ = [
    "APPROVED_RELATIONS",
    "MAX_DEFAULT_LENGTH",
    "PathResult",
    "build_adjacency",
    "reason_paths",
]
