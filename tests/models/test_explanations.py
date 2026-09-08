import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

from phenotype_network_v0.models.explanations import (
    enumerate_evidence_paths,
    remove_relation_edge,
)


def _matrix(size: int, records: list[tuple[int, int, float]]) -> csr_matrix:
    if not records:
        return csr_matrix((size, size), dtype=float)
    rows, columns, values = zip(*records, strict=True)
    return csr_matrix((values, (rows, columns)), shape=(size, size))


def _graph() -> tuple[pd.DataFrame, dict[str, csr_matrix]]:
    node_map = pd.DataFrame({
        "node_index": range(5),
        "node_type": ["phenotype", "phenotype", "gene", "gene", "pathway"],
        "node_id": ["HP:1", "HP:2", "10", "20", "R-HSA-1"],
    })
    relations = {
        "hpo_hpo": _matrix(5, [(0, 1, 0.8)]),
        "hpo_gene": _matrix(5, [(0, 2, 0.7), (1, 2, 0.6), (1, 3, 0.5)]),
        "gene_hpo": _matrix(5, [(2, 0, 0.7), (2, 1, 0.6), (3, 1, 0.5)]),
        "gene_gene": _matrix(5, [(2, 3, 0.9), (3, 2, 0.9)]),
        "gene_pathway": _matrix(5, [(2, 4, 0.5)]),
        "pathway_gene": _matrix(5, [(4, 3, 0.4)]),
        "pathway_pathway": csr_matrix((5, 5), dtype=float),
    }
    return node_map, relations


def test_allowed_paths_are_simple_deterministic_and_cost_sorted() -> None:
    node_map, relations = _graph()
    penalties = {name: 0.0 for name in relations}
    paths = enumerate_evidence_paths(
        ["HP:1"], "20", relations, node_map, penalties, maximum_paths=4
    )
    assert len(paths) == 4
    assert [path["path_cost"] for path in paths] == sorted(
        path["path_cost"] for path in paths
    )
    assert all(len(path["node_ids"]) == len(set(path["node_ids"])) for path in paths)
    assert {tuple(path["relations"]) for path in paths} == {
        ("hpo_hpo", "hpo_gene"),
        ("hpo_gene", "gene_gene"),
        ("hpo_gene", "gene_pathway", "pathway_gene"),
        ("hpo_hpo", "hpo_gene", "gene_gene"),
    }
    expected = -np.log(0.7) - np.log(0.9)
    assert paths[0]["path_cost"] == expected


def test_relation_edge_removal_updates_reverse_without_mutating_input() -> None:
    _, relations = _graph()
    updated = remove_relation_edge(relations, "hpo_gene", 0, 2)
    assert relations["hpo_gene"][0, 2] == 0.7
    assert updated["hpo_gene"][0, 2] == 0
    assert updated["gene_hpo"][2, 0] == 0