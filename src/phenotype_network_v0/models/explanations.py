"""Evidence-constrained path enumeration and relation edge deletion."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

ALLOWED_TEMPLATES = (
    ("hpo_hpo", "hpo_gene"),
    ("hpo_gene", "gene_gene"),
    ("hpo_gene", "gene_pathway", "pathway_gene"),
    ("hpo_hpo", "hpo_gene", "gene_gene"),
)
RELATION_NODE_TYPES = {
    "hpo_hpo": ("phenotype", "phenotype"),
    "hpo_gene": ("phenotype", "gene"),
    "gene_hpo": ("gene", "phenotype"),
    "gene_gene": ("gene", "gene"),
    "gene_pathway": ("gene", "pathway"),
    "pathway_gene": ("pathway", "gene"),
    "pathway_pathway": ("pathway", "pathway"),
}
REVERSE_RELATIONS = {
    "hpo_gene": "gene_hpo",
    "gene_hpo": "hpo_gene",
    "gene_pathway": "pathway_gene",
    "pathway_gene": "gene_pathway",
    "hpo_hpo": "hpo_hpo",
    "gene_gene": "gene_gene",
    "pathway_pathway": "pathway_pathway",
}


def enumerate_evidence_paths(
    source_hpo_ids: Sequence[str],
    target_gene_id: str,
    relations: Mapping[str, csr_matrix],
    node_map: pd.DataFrame,
    relation_penalties: Mapping[str, float],
    *,
    maximum_paths: int = 3,
    epsilon: float = 1e-12,
) -> list[dict[str, object]]:
    """Enumerate the lowest-cost allowed simple paths to one target gene."""
    if maximum_paths < 1 or epsilon <= 0:
        raise ValueError("Invalid path enumeration settings")
    index = {
        (str(row.node_type), str(row.node_id)): int(row.node_index)
        for row in node_map.itertuples(index=False)
    }
    node_type = node_map.set_index("node_index")["node_type"].astype(str).to_dict()
    node_id = node_map.set_index("node_index")["node_id"].astype(str).to_dict()
    target = index.get(("gene", str(target_gene_id)))
    if target is None:
        raise ValueError(f"Target gene is absent from node map: {target_gene_id}")
    sources = [
        index[("phenotype", str(hpo_id))]
        for hpo_id in source_hpo_ids
        if ("phenotype", str(hpo_id)) in index
    ]
    candidates: dict[tuple[int, ...], dict[str, object]] = {}
    for template in ALLOWED_TEMPLATES:
        if any(relation not in relations for relation in template):
            raise ValueError(f"Missing relation matrix for template: {template}")
        if any(relation not in relation_penalties for relation in template):
            raise ValueError(f"Missing relation penalty for template: {template}")
        for source in sources:
            _walk_template(
                source,
                target,
                template,
                relations,
                node_type,
                node_id,
                relation_penalties,
                epsilon,
                candidates,
            )
    return sorted(
        candidates.values(),
        key=lambda value: (
            float(value["path_cost"]),
            tuple(value["node_ids"]),
            tuple(value["relations"]),
        ),
    )[:maximum_paths]


def _walk_template(
    source: int,
    target: int,
    template: tuple[str, ...],
    relations: Mapping[str, csr_matrix],
    node_type: dict[int, str],
    node_id: dict[int, str],
    penalties: Mapping[str, float],
    epsilon: float,
    candidates: dict[tuple[int, ...], dict[str, object]],
) -> None:
    states: list[tuple[tuple[int, ...], tuple[float, ...]]] = [((source,), ())]
    for position, relation in enumerate(template):
        expected_source, expected_target = RELATION_NODE_TYPES[relation]
        final = position == len(template) - 1
        next_states = []
        matrix = relations[relation].tocsr()
        for nodes, weights in states:
            current = nodes[-1]
            if node_type[current] != expected_source:
                continue
            if final:
                weight = float(matrix[current, target])
                neighbor_values = ((target, weight),) if weight > 0 else ()
            else:
                row = matrix.getrow(current)
                neighbor_values = zip(row.indices, row.data, strict=True)
            for neighbor, weight in neighbor_values:
                neighbor = int(neighbor)
                weight = float(weight)
                if weight <= 0 or neighbor in nodes:
                    continue
                if node_type.get(neighbor) != expected_target:
                    continue
                next_states.append((nodes + (neighbor,), weights + (weight,)))
        states = next_states
        if not states:
            break
    for nodes, weights in states:
        if nodes[-1] != target:
            continue
        cost = sum(
            -np.log(max(weight, epsilon)) + float(penalties[relation])
            for relation, weight in zip(template, weights, strict=True)
        )
        record = {
            "node_indices": list(nodes),
            "node_ids": [node_id[value] for value in nodes],
            "node_types": [node_type[value] for value in nodes],
            "relations": list(template),
            "edge_weights": list(weights),
            "minimum_edge_weight": min(weights),
            "path_cost": float(cost),
            "path_score": float(np.exp(-cost)),
        }
        previous = candidates.get(nodes)
        if previous is None or float(record["path_cost"]) < float(previous["path_cost"]):
            candidates[nodes] = record


def remove_relation_edge(
    relations: Mapping[str, csr_matrix],
    relation: str,
    source_index: int,
    target_index: int,
) -> dict[str, csr_matrix]:
    """Remove an edge and its canonical reverse representation."""
    if relation not in relations or relation not in REVERSE_RELATIONS:
        raise ValueError(f"Unknown relation: {relation}")
    updated = {name: matrix.tocsr() for name, matrix in relations.items()}
    forward = updated[relation].tolil(copy=True)
    forward[source_index, target_index] = 0.0
    updated[relation] = forward.tocsr()
    updated[relation].eliminate_zeros()
    reverse_relation = REVERSE_RELATIONS[relation]
    reverse = updated[reverse_relation].tolil(copy=True)
    reverse[target_index, source_index] = 0.0
    updated[reverse_relation] = reverse.tocsr()
    updated[reverse_relation].eliminate_zeros()
    return updated