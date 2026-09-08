"""Tests for the V1 leakage audit and PyG graph conversion (Milestone 1).

Positive and negative fixtures exercise every leakage check plus the
deterministic graph serialization and V0 node-index roundtrip requirements.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from phenotype_network_v1.data.leakage import run_fold_leakage
from phenotype_network_v1.data.pyg_graph import build_heterodata, sha256_heterodata
from phenotype_network_v1.data.query_dataset import (
    build_fold_queries,
    write_query_instances,
)

NODE_MAP = pd.DataFrame(
    {
        "node_index": [0, 1, 2],
        "node_type": ["phenotype", "phenotype", "gene"],
        "node_id": ["HP:1", "HP:2", "100"],
        "node_name": ["a", "b", "gene-100"],
    }
)


def _write_parquet(path, frame) -> None:  # noqa: ANN001
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)


def _save_csr(path, rows, cols, values, size=3) -> None:  # noqa: ANN001
    path.parent.mkdir(parents=True, exist_ok=True)
    matrix = sp.csr_matrix((values, (rows, cols)), shape=(size, size))
    sp.save_npz(path, matrix)


def make_layout(tmp_path, *, extra_test_disease: bool = False) -> dict:  # noqa: ANN001
    root = tmp_path
    processed = root / "processed"
    folds = root / "folds"
    graphs = root / "graphs"
    fold_dir = folds / "fold_0"
    graph_dir = graphs / "fold_0"
    fold_dir.mkdir(parents=True, exist_ok=True)
    graph_dir.mkdir(parents=True, exist_ok=True)

    _write_parquet(
        processed / "cardiovascular_disease_phenotypes.parquet",
        pd.DataFrame(
            {
                "disease_id": ["d1", "d2"],
                "hpo_id": ["HP:1", "HP:2"],
                "frequency_weight": [1.0, 0.5],
            }
        ),
    )
    _write_parquet(
        processed / "cardiovascular_disease_genes.parquet",
        pd.DataFrame({"disease_id": ["d1", "d2"], "gene_id": ["100", "200"]}),
    )
    _write_parquet(
        folds / "disease_fold_assignments.parquet",
        pd.DataFrame(
            {
                "disease_id": ["d1", "d2"],
                "disease_family_id": ["FAM-A", "FAM-B"],
                "fold": [0, 0],
            }
        ),
    )
    (fold_dir / "train_diseases.txt").write_text("d1\n", encoding="utf-8")
    (fold_dir / "val_diseases.txt").write_text("", encoding="utf-8")
    (fold_dir / "test_diseases.txt").write_text("d2\n", encoding="utf-8")
    _write_parquet(
        fold_dir / "train_hpo_ic.parquet",
        pd.DataFrame(
            {"hpo_id": ["HP:1", "HP:2"], "ic": [2.0, 1.0]}
        ),
    )
    _write_parquet(
        fold_dir / "candidate_gene_universe.parquet",
        pd.DataFrame({"gene_id": ["100", "200"]}),
    )
    _write_parquet(fold_dir / "phenotype_gene_edges.parquet", pd.DataFrame())
    (fold_dir / "leakage_report.json").write_text(
        json.dumps(
            {
                "fold": 0,
                "checks": {
                    "hpo_gene_edges_train_only": True,
                    "no_test_contributors": True,
                    "no_validation_contributors": True,
                    "no_direct_answer_edges": True,
                    "family_sets_disjoint": True,
                },
            }
        ),
        encoding="utf-8",
    )

    graph_dir.mkdir(parents=True, exist_ok=True)
    NODE_MAP.to_parquet(graph_dir / "node_map.parquet", index=False)
    (graph_dir / "graph_qc.json").write_text('{"maximum_row_sum_error": 0.0}\n')
    # phenotype -> phenotype edge (0->1) and gene -> phenotype edge (2->0).
    _save_csr(graph_dir / "A_hpo_hpo.npz", rows=[0], cols=[1], values=[0.5])
    _save_csr(graph_dir / "A_gene_hpo.npz", rows=[2], cols=[0], values=[0.25])

    # Build the real query parquet through the production path.
    queries = build_fold_queries(
        0,
        processed_dir=processed,
        folds_dir=folds,
        assignments_path=folds / "disease_fold_assignments.parquet",
    )
    v1_processed = root / "data" / "processed" / "v1" / "fold_0"
    queries_path = v1_processed / "query_instances.parquet"
    write_query_instances(queries, queries_path)

    v1_graphs = root / "data" / "graphs" / "v1" / "fold_0"
    manifest_path = v1_graphs / "graph_manifest.json"
    v1_graphs.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(
            {
                "fold": 0,
                "node_counts": {"phenotype": 2, "gene": 1, "pathway": 0},
                "edge_counts": {"edge_hpo_hpo": 1, "edge_gene_hpo": 1},
                "total_nodes": 3,
                "serialization_sha256": "a" * 64,
            }
        ),
        encoding="utf-8",
    )
    report_path = v1_graphs / "leakage_report.json"
    return {
        "root": root,
        "processed": processed,
        "folds": folds,
        "fold_dir": fold_dir,
        "graph_dir": graph_dir,
        "queries_path": queries_path,
        "manifest_path": manifest_path,
        "report_path": report_path,
        "v1_graphs": v1_graphs,
    }


def _run(layout) -> dict:  # noqa: ANN001
    return run_fold_leakage(
        0,
        processed_dir=layout["processed"],
        folds_dir=layout["folds"],
        queries_path=layout["queries_path"],
        manifest_path=layout["manifest_path"],
        report_path=layout["report_path"],
    )


def test_clean_layout_passes_all_checks(tmp_path) -> None:  # noqa: ANN001
    layout = make_layout(tmp_path)
    report = _run(layout)
    assert report["status"] == "PASS"
    assert report["passed_checks"] == report["failed_checks"] + report["passed_checks"]
    assert report["failed_checks"] == 0
    assert layout["report_path"].exists()


def test_split_mismatch_fails(tmp_path) -> None:  # noqa: ANN001
    layout = make_layout(tmp_path)
    # Introduce a disease into the frozen test file that no query row covers.
    with open(layout["fold_dir"] / "test_diseases.txt", "a", encoding="utf-8") as handle:
        handle.write("d-ghost\n")
    report = _run(layout)
    assert report["status"] == "FAIL"
    assert report["checks"]["split_partitions_match_frozen"]["passed"] is False


def test_edge_reconcile_mismatch_fails(tmp_path) -> None:  # noqa: ANN001
    layout = make_layout(tmp_path)
    manifest = json.loads(layout["manifest_path"].read_text(encoding="utf-8"))
    manifest["edge_counts"]["edge_hpo_hpo"] = 99  # does not match the npz
    layout["manifest_path"].write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    report = _run(layout)
    assert report["status"] == "FAIL"
    assert report["checks"]["graph_edge_counts_reconcile_v0"]["passed"] is False


def test_hpo_outside_training_ic_fails(tmp_path) -> None:  # noqa: ANN001
    layout = make_layout(tmp_path)
    ic_path = layout["fold_dir"] / "train_hpo_ic.parquet"
    reduced = pd.read_parquet(ic_path)
    reduced = reduced[reduced["hpo_id"] != "HP:2"]
    reduced.to_parquet(ic_path, index=False)
    report = _run(layout)
    assert report["status"] == "FAIL"
    assert report["checks"]["hpo_ids_in_fold_training_ic"]["passed"] is False


def test_positive_gene_outside_universe_fails(tmp_path) -> None:  # noqa: ANN001
    layout = make_layout(tmp_path)
    universe_path = layout["fold_dir"] / "candidate_gene_universe.parquet"
    pd.DataFrame({"gene_id": ["999"]}).to_parquet(universe_path, index=False)
    report = _run(layout)
    assert report["status"] == "FAIL"
    assert report["checks"]["positive_genes_covered_or_unresolved"]["passed"] is False


def test_heterodata_deterministic_serialization(tmp_path) -> None:  # noqa: ANN001
    layout = make_layout(tmp_path)
    first = build_heterodata(layout["graph_dir"])
    second = build_heterodata(layout["graph_dir"])
    assert sha256_heterodata(first) == sha256_heterodata(second)
    assert first["phenotype"].num_nodes == 2
    assert first["gene"].num_nodes == 1
    assert ("gene", "gene_hpo", "phenotype") in first.edge_types
    assert first["gene", "gene_hpo", "phenotype"].edge_index.size(1) == 1


def test_v0_node_index_roundtrip(tmp_path) -> None:  # noqa: ANN001
    layout = make_layout(tmp_path)
    data = build_heterodata(layout["graph_dir"])
    # Gene local node 0 must carry global index 2 from the fixture node_map.
    assert data["gene"].v0_node_index.tolist() == [2]
    assert data["phenotype"].v0_node_index.tolist() == [0, 1]


def test_no_cuda_required_for_graph_build(tmp_path) -> None:  # noqa: ANN001
    layout = make_layout(tmp_path)
    data = build_heterodata(layout["graph_dir"])
    assert int(data["phenotype"].num_nodes) >= 0
