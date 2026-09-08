"""Build fold-specific sparse heterogeneous transition matrices."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, diags, eye, save_npz

RELATION_BUDGET_KEYS = {
    "hpo_hpo": ("phenotype", "phenotype_is_a_phenotype"),
    "hpo_gene": ("phenotype", "phenotype_associated_with_gene"),
    "gene_hpo": ("gene", "gene_associated_with_phenotype"),
    "gene_gene": ("gene", "gene_interacts_with_gene"),
    "gene_pathway": ("gene", "gene_participates_in_pathway"),
    "pathway_gene": ("pathway", "pathway_contains_gene"),
    "pathway_pathway": ("pathway", "pathway_part_of_pathway"),
}


@dataclass(frozen=True)
class GraphResult:
    """Node map, relation matrices, transition matrix, and QC."""

    node_map: pd.DataFrame
    relations: dict[str, csr_matrix]
    transition: csr_matrix
    qc: dict[str, Any]


def build_heterogeneous_graph(
    hpo_nodes: pd.DataFrame,
    hpo_edges: pd.DataFrame,
    candidates: pd.DataFrame,
    hpo_gene_edges: pd.DataFrame,
    string_edges: pd.DataFrame,
    reactome_gene_pathway: pd.DataFrame,
    pathways: pd.DataFrame,
    pathway_hierarchy: pd.DataFrame,
    relation_budget: dict[str, dict[str, float]],
) -> GraphResult:
    """Build one fold graph and a row-stochastic transition matrix."""
    node_map = _node_map(hpo_nodes, candidates, pathways)
    index = {(row.node_type, str(row.node_id)): int(row.node_index) for row in node_map.itertuples(index=False)}
    size = len(node_map)
    hpo_pairs = []
    for row in hpo_edges.itertuples(index=False):
        child, parent = index[("phenotype", str(row.child_hpo_id))], index[("phenotype", str(row.parent_hpo_id))]
        hpo_pairs.extend([(child, parent, 1.0), (parent, child, 1.0)])
    hpo_gene_pairs = [
        (index[("phenotype", str(row.hpo_id))], index[("gene", str(row.gene_id))], float(row.weight))
        for row in hpo_gene_edges.itertuples(index=False)
        if ("phenotype", str(row.hpo_id)) in index and ("gene", str(row.gene_id)) in index
    ]
    gene_gene_pairs = []
    for row in string_edges.itertuples(index=False):
        a, b = index[("gene", str(row.gene_a))], index[("gene", str(row.gene_b))]
        weight = float(row.weight)
        gene_gene_pairs.extend([(a, b, weight), (b, a, weight)])
    gene_pathway_pairs = [
        (index[("gene", str(row.gene_id))], index[("pathway", str(row.pathway_id))], float(row.weight))
        for row in reactome_gene_pathway.itertuples(index=False)
        if ("gene", str(row.gene_id)) in index and ("pathway", str(row.pathway_id)) in index
    ]
    pathway_pairs = []
    for row in pathway_hierarchy.itertuples(index=False):
        parent_key, child_key = ("pathway", str(row.parent_pathway_id)), ("pathway", str(row.child_pathway_id))
        if parent_key in index and child_key in index:
            parent, child = index[parent_key], index[child_key]
            pathway_pairs.extend([(parent, child, 1.0), (child, parent, 1.0)])
    relations = {
        "hpo_hpo": _matrix(hpo_pairs, size),
        "hpo_gene": _matrix(hpo_gene_pairs, size),
        "gene_hpo": _matrix([(b, a, w) for a, b, w in hpo_gene_pairs], size),
        "gene_gene": _matrix(gene_gene_pairs, size),
        "gene_pathway": _matrix(gene_pathway_pairs, size),
        "pathway_gene": _matrix([(b, a, w) for a, b, w in gene_pathway_pairs], size),
        "pathway_pathway": _matrix(pathway_pairs, size),
    }
    transition = compose_transition_matrix(relations, relation_budget)
    relation_degree = np.zeros(size, dtype=np.int64)
    for matrix in relations.values():
        relation_degree += np.asarray(matrix.getnnz(axis=1)).ravel()
    isolated = np.flatnonzero(relation_degree == 0)
    row_sums = np.asarray(transition.sum(axis=1)).ravel()
    relation_hashes = {name: _sparse_hash(matrix) for name, matrix in relations.items()}
    qc = {
        "nodes": len(node_map),
        "node_counts": node_map.groupby("node_type").size().to_dict(),
        "relation_nnz": {name: int(matrix.nnz) for name, matrix in relations.items()},
        "relation_sha256": relation_hashes,
        "matrix_shape": [size, size],
        "transition_nnz": int(transition.nnz),
        "transition_sha256": _sparse_hash(transition),
        "isolated_nodes_with_self_loop": len(isolated),
        "negative_relation_weights": int(sum((matrix.data < 0).sum() for matrix in relations.values())),
        "row_sum_min": float(row_sums.min()) if len(row_sums) else None,
        "row_sum_max": float(row_sums.max()) if len(row_sums) else None,
        "maximum_row_sum_error": float(np.max(np.abs(row_sums - 1.0))) if len(row_sums) else None,
        "candidate_genes": int((node_map["node_type"] == "gene").sum()),
        "hpo_gene_seed_terms": int(hpo_gene_edges["hpo_id"].nunique()) if not hpo_gene_edges.empty else 0,
    }
    if qc["negative_relation_weights"] or (qc["maximum_row_sum_error"] or 0) > 1e-10:
        raise ValueError("Graph QC failed: negative weights or non-stochastic transition rows")
    return GraphResult(node_map, relations, transition.tocsr(), qc)


def compose_transition_matrix(
    relations: dict[str, csr_matrix],
    relation_budget: dict[str, dict[str, float]],
) -> csr_matrix:
    """Compose a row-stochastic transition matrix from selected relation matrices."""
    required = set(RELATION_BUDGET_KEYS)
    missing = required - set(relations)
    if missing:
        raise ValueError(f"Relation matrices are missing: {', '.join(sorted(missing))}")
    shapes = {relations[name].shape for name in required}
    if len(shapes) != 1:
        raise ValueError("All relation matrices must have the same shape")
    shape = shapes.pop()
    if shape[0] != shape[1]:
        raise ValueError("Relation matrices must be square")
    if any(
        matrix.nnz and (not np.isfinite(matrix.data).all() or np.any(matrix.data < 0))
        for matrix in relations.values()
    ):
        raise ValueError("Relation matrices must contain finite nonnegative weights")
    size = shape[0]
    budgets = _validate_budgets(relation_budget)
    normalized = {
        name: _row_normalize(relations[name].tocsr())
        for name in required
    }
    available_budget = np.zeros(size, dtype=float)
    row_available: dict[str, np.ndarray] = {}
    for name, matrix in normalized.items():
        present = np.asarray(matrix.getnnz(axis=1) > 0, dtype=float)
        row_available[name] = present
        available_budget += present * budgets[name]
    transition = csr_matrix(shape, dtype=float)
    for name, matrix in normalized.items():
        scale = np.divide(
            budgets[name],
            available_budget,
            out=np.zeros(size),
            where=available_budget > 0,
        )
        transition = transition + diags(scale * row_available[name]) @ matrix
    isolated = np.flatnonzero(available_budget == 0)
    if len(isolated):
        transition = transition + csr_matrix(
            (np.ones(len(isolated)), (isolated, isolated)),
            shape=shape,
        )
    transition.eliminate_zeros()
    row_sums = np.asarray(transition.sum(axis=1)).ravel()
    if not np.allclose(row_sums, 1.0, atol=1e-12, rtol=1e-12):
        raise ValueError("Composed transition matrix is not row-stochastic")
    return transition.tocsr()


def write_graph_outputs(result: GraphResult, graph_dir: Path, *, overwrite: bool = False) -> None:
    """Write node, relation, transition, and QC artifacts for one fold."""
    folder = Path(graph_dir)
    destinations = [folder / "node_map.parquet", folder / "transition_matrix.npz", folder / "graph_qc.json"]
    destinations.extend(folder / f"A_{name}.npz" for name in result.relations)
    existing = [path for path in destinations if path.exists()]
    if existing and not overwrite:
        raise FileExistsError("Graph outputs already exist; use overwrite=True: " + ", ".join(map(str, existing)))
    folder.mkdir(parents=True, exist_ok=True)
    result.node_map.to_parquet(folder / "node_map.parquet", index=False)
    for name, matrix in result.relations.items():
        save_npz(folder / f"A_{name}.npz", matrix)
    save_npz(folder / "transition_matrix.npz", result.transition)
    (folder / "graph_qc.json").write_text(json.dumps(result.qc, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _node_map(hpo_nodes: pd.DataFrame, candidates: pd.DataFrame, pathways: pd.DataFrame) -> pd.DataFrame:
    hpo = hpo_nodes[["hpo_id", "hpo_name"]].rename(columns={"hpo_id": "node_id", "hpo_name": "node_name"})
    hpo["node_type"] = "phenotype"
    genes = pd.DataFrame({"node_id": sorted(candidates["gene_id"].astype(str), key=int)})
    genes["node_name"] = ""
    genes["node_type"] = "gene"
    selected_pathways = pathways.loc[pathways["included_in_primary_graph"], ["pathway_id", "pathway_name"]].rename(
        columns={"pathway_id": "node_id", "pathway_name": "node_name"}
    ).sort_values("node_id", kind="stable")
    selected_pathways["node_type"] = "pathway"
    frame = pd.concat([hpo.sort_values("node_id"), genes, selected_pathways], ignore_index=True)
    if frame.duplicated(["node_type", "node_id"]).any():
        raise ValueError("Node identifiers are not unique within type")
    frame.insert(0, "node_index", np.arange(len(frame), dtype=np.int64))
    return frame[["node_index", "node_type", "node_id", "node_name"]]


def _matrix(records: list[tuple[int, int, float]], size: int) -> csr_matrix:
    if not records:
        return csr_matrix((size, size), dtype=float)
    rows, columns, values = zip(*records, strict=True)
    matrix = csr_matrix((values, (rows, columns)), shape=(size, size), dtype=float)
    matrix.sum_duplicates()
    matrix.eliminate_zeros()
    return matrix


def _row_normalize(matrix: csr_matrix) -> csr_matrix:
    row_sums = np.asarray(matrix.sum(axis=1)).ravel()
    inverse = np.divide(1.0, row_sums, out=np.zeros_like(row_sums, dtype=float), where=row_sums > 0)
    return (diags(inverse) @ matrix).tocsr()


def _validate_budgets(config: dict[str, dict[str, float]]) -> dict[str, float]:
    budgets = {}
    for relation, (source_type, key) in RELATION_BUDGET_KEYS.items():
        value = float(config[source_type][key])
        if value < 0:
            raise ValueError(f"Negative relation budget: {relation}")
        budgets[relation] = value
    for source_type in ("phenotype", "gene", "pathway"):
        total = sum(budgets[name] for name, (owner, _) in RELATION_BUDGET_KEYS.items() if owner == source_type)
        if not np.isclose(total, 1.0):
            raise ValueError(f"Relation budgets for {source_type} must sum to one; got {total}")
    return budgets


def _sparse_hash(matrix: csr_matrix) -> str:
    matrix = matrix.tocsr()
    digest = hashlib.sha256()
    digest.update(np.asarray(matrix.shape, dtype=np.int64).tobytes())
    digest.update(matrix.indptr.astype(np.int64).tobytes())
    digest.update(matrix.indices.astype(np.int64).tobytes())
    digest.update(matrix.data.astype(np.float64).tobytes())
    return digest.hexdigest()