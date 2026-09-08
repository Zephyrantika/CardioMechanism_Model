"""Path reasoning tests (Milestone 6)."""

from __future__ import annotations

import pytest

from phenotype_network_v1.models.path_reasoner import (
    APPROVED_RELATIONS,
    build_adjacency,
    reason_paths,
)


def _graph():
    edge_index = __import__("torch").tensor(
        [
            # HPO 0 -> HPO 1 -> gene 3; HPO 0 -> gene 3 direct; gene 3 -> gene 4.
            [0, 1, 0, 3],
            [1, 3, 3, 4],
        ],
        dtype=__import__("torch").long,
    )
    edge_type = __import__("torch").tensor([0, 0, 1, 2], dtype=__import__("torch").long)
    return build_adjacency(
        edge_index, edge_type, ["hpo_hpo", "hpo_gene", "gene_gene"]
    )


def test_approved_relations_filter() -> None:
    assert "gene_hpo" in APPROVED_RELATIONS
    assert len(APPROVED_RELATIONS) == 7


def test_paths_reach_gene_targets() -> None:
    adjacency = _graph()
    paths = reason_paths(adjacency, [0], {3}, max_length=3, max_paths=10)
    assert paths
    for path in paths:
        assert path.nodes[0] == 0
        assert path.nodes[-1] == 3
        assert len(path.relations) == len(path.nodes) - 1


def test_paths_are_cycle_free() -> None:
    adjacency = _graph()
    paths = reason_paths(adjacency, [0], {4}, max_length=5, max_paths=50)
    for path in paths:
        assert len(path.nodes) == len(set(path.nodes))
        assert path.nodes[-1] == 4


def test_no_path_when_target_disconnected() -> None:
    adjacency = _graph()
    paths = reason_paths(adjacency, [0], {99}, max_length=3)
    assert paths == []


def test_invalid_length_rejected() -> None:
    adjacency = _graph()
    with pytest.raises(ValueError):
        reason_paths(adjacency, [0], {3}, max_length=0)
