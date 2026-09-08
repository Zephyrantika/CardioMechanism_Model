"""Sparse relation substitutions for graph ablation experiments."""

from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

ALL_RELATIONS = frozenset({
    "hpo_hpo",
    "hpo_gene",
    "gene_hpo",
    "gene_gene",
    "gene_pathway",
    "pathway_gene",
    "pathway_pathway",
})

VARIANT_ACTIVE_RELATIONS = {
    "biogrid": ALL_RELATIONS,
    "hpo_gene_only": frozenset({"hpo_gene", "gene_hpo"}),
    "hpo_gene_ppi": frozenset({"hpo_gene", "gene_hpo", "gene_gene"}),
    "no_hpo_hierarchy": ALL_RELATIONS - {"hpo_hpo"},
    "no_ppi": ALL_RELATIONS - {"gene_gene"},
    "no_reactome": ALL_RELATIONS - {
        "gene_pathway", "pathway_gene", "pathway_pathway",
    },
    "no_frequency": ALL_RELATIONS,
    "no_ic": ALL_RELATIONS,
    "string900": ALL_RELATIONS,
}


def select_active_relations(
    relations: dict[str, csr_matrix],
    active_relations: frozenset[str],
) -> dict[str, csr_matrix]:
    """Return all relation keys, replacing disabled matrices with sparse zeros."""
    if set(relations) != ALL_RELATIONS:
        raise ValueError("Relation dictionary does not contain the canonical relation set")
    unknown = set(active_relations) - ALL_RELATIONS
    if unknown:
        raise ValueError(f"Unknown active relations: {', '.join(sorted(unknown))}")
    shape = next(iter(relations.values())).shape
    return {
        name: relations[name].tocsr()
        if name in active_relations
        else csr_matrix(shape, dtype=float)
        for name in sorted(ALL_RELATIONS)
    }


def hpo_gene_relation_matrices(
    edges: pd.DataFrame,
    node_map: pd.DataFrame,
) -> tuple[csr_matrix, csr_matrix, pd.DataFrame]:
    """Build HPO-gene matrices and report edges excluded from the node map."""
    index = {
        (str(row.node_type), str(row.node_id)): int(row.node_index)
        for row in node_map.itertuples(index=False)
    }
    size = len(node_map)
    records = []
    unresolved: list[dict[str, str]] = []
    for row in edges.itertuples(index=False):
        hpo_key = ("phenotype", str(row.hpo_id))
        gene_key = ("gene", str(row.gene_id))
        if hpo_key not in index or gene_key not in index:
            unresolved.append({
                "hpo_id": str(row.hpo_id),
                "gene_id": str(row.gene_id),
                "reason": (
                    "missing_hpo_node"
                    if hpo_key not in index
                    else "gene_outside_candidate_universe"
                ),
            })
            continue
        records.append((index[hpo_key], index[gene_key], float(row.weight)))
    forward = _matrix(records, size)
    reverse = _matrix([(target, source, weight) for source, target, weight in records], size)
    audit = pd.DataFrame.from_records(
        unresolved, columns=["hpo_id", "gene_id", "reason"]
    )
    return forward, reverse, audit


def ppi_relation_matrix(
    edges: pd.DataFrame,
    node_map: pd.DataFrame,
) -> csr_matrix:
    """Build a symmetric full-shape PPI relation matrix."""
    index = {
        str(row.node_id): int(row.node_index)
        for row in node_map.loc[node_map["node_type"].eq("gene")].itertuples(index=False)
    }
    size = len(node_map)
    records = []
    unresolved = []
    for row in edges.itertuples(index=False):
        left, right = str(row.gene_a), str(row.gene_b)
        if left not in index or right not in index:
            unresolved.append((left, right))
            continue
        weight = float(row.weight)
        records.extend([
            (index[left], index[right], weight),
            (index[right], index[left], weight),
        ])
    if unresolved:
        raise ValueError(f"PPI ablation edges contain unresolved genes: {unresolved[:5]}")
    return _matrix(records, size)


def sparse_matrix_sha256(matrix: csr_matrix) -> str:
    """Hash a CSR matrix including shape, structure, and values."""
    value = matrix.tocsr()
    digest = hashlib.sha256()
    digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
    digest.update(value.indptr.astype(np.int64).tobytes())
    digest.update(value.indices.astype(np.int64).tobytes())
    digest.update(value.data.astype(np.float64).tobytes())
    return digest.hexdigest()


def _matrix(records: list[tuple[int, int, float]], size: int) -> csr_matrix:
    if not records:
        return csr_matrix((size, size), dtype=float)
    rows, columns, values = zip(*records, strict=True)
    matrix = csr_matrix(
        (values, (rows, columns)), shape=(size, size), dtype=float
    )
    matrix.sum_duplicates()
    matrix.eliminate_zeros()
    if matrix.nnz and (
        not np.isfinite(matrix.data).all() or np.any(matrix.data < 0)
    ):
        raise ValueError("Ablation relation matrix contains invalid weights")
    return matrix
