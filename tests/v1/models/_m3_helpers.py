"""Synthetic shared graph helpers for M3 model tests."""

from __future__ import annotations

import torch


def small_graph() -> dict:
    """5 phenotype (0..4), 4 gene (5..8), hpo->hpo, hpo->gene, gene->gene."""
    edge_index = torch.tensor(
        [
            [0, 1, 2, 3],
            [1, 2, 5, 6],
        ],
        dtype=torch.long,
    )
    edge_type = torch.tensor([0, 0, 1, 2], dtype=torch.long)
    return {
        "num_nodes": 9,
        "edge_index": edge_index,
        "edge_type": edge_type,
        "num_relations": 3,
        "relation_names": ["hpo_hpo", "hpo_gene", "gene_gene"],
    }


def gene_node_indices() -> list[int]:
    return [5, 6, 7, 8]


def hpo_offset() -> int:
    return 0
