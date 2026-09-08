import numpy as np
import pandas as pd
import pytest
from scipy.sparse import csr_matrix

from phenotype_network_v0.evaluation.ablation import (
    ALL_RELATIONS,
    hpo_gene_relation_matrices,
    ppi_relation_matrix,
    select_active_relations,
    sparse_matrix_sha256,
)


def _node_map() -> pd.DataFrame:
    return pd.DataFrame({
        "node_index": [0, 1, 2, 3],
        "node_type": ["phenotype", "gene", "gene", "pathway"],
        "node_id": ["HP:1", "10", "20", "R-HSA-1"],
    })


def test_relation_selection_preserves_keys_and_zeros_disabled_relations() -> None:
    relations = {
        name: csr_matrix(np.eye(4))
        for name in ALL_RELATIONS
    }
    selected = select_active_relations(
        relations, frozenset({"hpo_gene", "gene_hpo"})
    )
    assert set(selected) == ALL_RELATIONS
    assert selected["hpo_gene"].nnz == 4
    assert selected["gene_hpo"].nnz == 4
    assert all(
        selected[name].nnz == 0
        for name in ALL_RELATIONS - {"hpo_gene", "gene_hpo"}
    )


def test_hpo_gene_and_ppi_relation_matrix_direction_and_hash() -> None:
    hpo_edges = pd.DataFrame({
        "hpo_id": ["HP:1"],
        "gene_id": ["10"],
        "weight": [0.75],
    })
    forward, reverse, audit = hpo_gene_relation_matrices(hpo_edges, _node_map())
    assert forward[0, 1] == pytest.approx(0.75)
    assert reverse[1, 0] == pytest.approx(0.75)
    assert audit.empty

    ppi_edges = pd.DataFrame({
        "gene_a": ["10"],
        "gene_b": ["20"],
        "weight": [0.9],
    })
    ppi = ppi_relation_matrix(ppi_edges, _node_map())
    assert ppi[1, 2] == pytest.approx(0.9)
    assert ppi[2, 1] == pytest.approx(0.9)
    assert sparse_matrix_sha256(ppi) == sparse_matrix_sha256(ppi.copy())


def test_hpo_gene_relation_builder_audits_unresolved_nodes() -> None:
    edges = pd.DataFrame({
        "hpo_id": ["HP:9", "HP:1"],
        "gene_id": ["10", "99"],
        "weight": [1.0, 1.0],
    })
    forward, reverse, audit = hpo_gene_relation_matrices(edges, _node_map())
    assert forward.nnz == 0
    assert reverse.nnz == 0
    assert audit.to_dict("records") == [
        {"hpo_id": "HP:9", "gene_id": "10", "reason": "missing_hpo_node"},
        {
            "hpo_id": "HP:1",
            "gene_id": "99",
            "reason": "gene_outside_candidate_universe",
        },
    ]


def test_ppi_relation_builder_rejects_unresolved_nodes() -> None:
    with pytest.raises(ValueError, match="unresolved"):
        ppi_relation_matrix(
            pd.DataFrame({"gene_a": ["10"], "gene_b": ["99"], "weight": [1.0]}),
            _node_map(),
        )
