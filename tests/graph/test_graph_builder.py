from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.sparse import load_npz

from phenotype_network_v0.graph.graph_builder import (
    build_heterogeneous_graph,
    compose_transition_matrix,
    write_graph_outputs,
)

BUDGET = {
    "phenotype": {"phenotype_is_a_phenotype": 0.4, "phenotype_associated_with_gene": 0.6},
    "gene": {"gene_associated_with_phenotype": 0.15, "gene_interacts_with_gene": 0.6,
             "gene_participates_in_pathway": 0.25},
    "pathway": {"pathway_contains_gene": 0.7, "pathway_part_of_pathway": 0.3},
}


def _inputs():
    hpo_nodes = pd.DataFrame({"hpo_id": ["HP:1", "HP:2"], "hpo_name": ["root", "child"]})
    hpo_edges = pd.DataFrame({"child_hpo_id": ["HP:2"], "parent_hpo_id": ["HP:1"], "relation_type": ["is_a"]})
    candidates = pd.DataFrame({"gene_id": ["1", "2", "3"]})
    hpo_gene = pd.DataFrame({"hpo_id": ["HP:2"], "gene_id": ["1"], "weight": [0.8]})
    string = pd.DataFrame({"gene_a": ["1"], "gene_b": ["2"], "weight": [0.9]})
    reactome = pd.DataFrame({"gene_id": ["2"], "pathway_id": ["R-HSA-1"], "weight": [0.5]})
    pathways = pd.DataFrame({"pathway_id": ["R-HSA-1"], "pathway_name": ["path"],
                             "included_in_primary_graph": [True]})
    hierarchy = pd.DataFrame(columns=["parent_pathway_id", "child_pathway_id"])
    return hpo_nodes, hpo_edges, candidates, hpo_gene, string, reactome, pathways, hierarchy, BUDGET


def test_relation_dimensions_indices_and_normalization() -> None:
    result = build_heterogeneous_graph(*_inputs())
    assert result.node_map["node_index"].tolist() == list(range(6))
    assert not result.node_map.duplicated(["node_type", "node_id"]).any()
    assert all(matrix.shape == (6, 6) for matrix in result.relations.values())
    assert result.transition.shape == (6, 6)
    assert np.all(result.transition.data >= 0)
    assert np.allclose(np.asarray(result.transition.sum(axis=1)).ravel(), 1.0)


def test_transition_composition_roundtrip_and_relation_ablation() -> None:
    result = build_heterogeneous_graph(*_inputs())
    recomposed = compose_transition_matrix(result.relations, BUDGET)
    np.testing.assert_array_equal(recomposed.indptr, result.transition.indptr)
    np.testing.assert_array_equal(recomposed.indices, result.transition.indices)
    np.testing.assert_allclose(recomposed.data, result.transition.data)

    ablated = dict(result.relations)
    ablated["gene_gene"] = ablated["gene_gene"].__class__(ablated["gene_gene"].shape)
    transition = compose_transition_matrix(ablated, BUDGET)
    assert np.allclose(np.asarray(transition.sum(axis=1)).ravel(), 1.0)
    assert np.all(transition.data >= 0)
    assert (transition != result.transition).nnz > 0


def test_missing_relations_redistribute_budget_and_isolated_self_loops() -> None:
    result = build_heterogeneous_graph(*_inputs())
    index = {(row.node_type, row.node_id): row.node_index for row in result.node_map.itertuples()}
    gene1, gene2, gene3 = (index[("gene", value)] for value in ("1", "2", "3"))
    phenotype2 = index[("phenotype", "HP:2")]
    pathway = index[("pathway", "R-HSA-1")]
    assert result.transition[gene1, phenotype2] == pytest.approx(0.2)
    assert result.transition[gene1, gene2] == pytest.approx(0.8)
    assert result.transition[pathway, gene2] == pytest.approx(1.0)
    assert result.transition[gene3, gene3] == pytest.approx(1.0)
    assert result.qc["isolated_nodes_with_self_loop"] == 1


def test_only_train_derived_relation_changes_between_folds() -> None:
    inputs = list(_inputs())
    first = build_heterogeneous_graph(*inputs)
    inputs[3] = pd.DataFrame({"hpo_id": ["HP:1"], "gene_id": ["2"], "weight": [0.4]})
    second = build_heterogeneous_graph(*inputs)
    assert first.qc["relation_sha256"]["hpo_gene"] != second.qc["relation_sha256"]["hpo_gene"]
    assert first.qc["relation_sha256"]["gene_hpo"] != second.qc["relation_sha256"]["gene_hpo"]
    for relation in {"hpo_hpo", "gene_gene", "gene_pathway", "pathway_gene", "pathway_pathway"}:
        assert first.qc["relation_sha256"][relation] == second.qc["relation_sha256"][relation]


def test_graph_outputs_roundtrip_and_overwrite(tmp_path: Path) -> None:
    result = build_heterogeneous_graph(*_inputs())
    write_graph_outputs(result, tmp_path)
    assert load_npz(tmp_path / "transition_matrix.npz").shape == (6, 6)
    assert len(pd.read_parquet(tmp_path / "node_map.parquet")) == 6
    with pytest.raises(FileExistsError, match="overwrite=True"):
        write_graph_outputs(result, tmp_path)
    write_graph_outputs(result, tmp_path, overwrite=True)