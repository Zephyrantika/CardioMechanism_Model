import json
from pathlib import Path

import pandas as pd
import pytest

from phenotype_network_v0.data.hpo_parser import (
    EDGE_COLUMNS,
    NODE_COLUMNS,
    HpoOutputPaths,
    parse_hpo_ontology,
    write_hpo_outputs,
)

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "hpo_small.obo"


@pytest.fixture
def parsed():
    return parse_hpo_ontology(FIXTURE)


def test_active_hpo_nodes_have_expected_schema(parsed) -> None:
    assert list(parsed.nodes.columns) == NODE_COLUMNS
    assert parsed.nodes["hpo_id"].tolist() == ["HP:0000001", "HP:0000002", "HP:0000003", "HP:0000004"]
    assert not parsed.nodes["is_obsolete"].any()
    child = parsed.nodes.set_index("hpo_id").loc["HP:0000002"]
    assert child["definition"] == "A child phenotype."
    assert child["synonyms"] == ["Child feature"]


def test_is_a_edges_are_unique_and_reference_valid_nodes(parsed) -> None:
    assert list(parsed.edges.columns) == EDGE_COLUMNS
    assert parsed.edges.to_dict("records") == [
        {
            "child_hpo_id": "HP:0000002",
            "parent_hpo_id": "HP:0000001",
            "relation_type": "is_a",
        },
        {
            "child_hpo_id": "HP:0000003",
            "parent_hpo_id": "HP:0000002",
            "relation_type": "is_a",
        },        {
            "child_hpo_id": "HP:0000004",
            "parent_hpo_id": "HP:0000002",
            "relation_type": "is_a",
        },
    ]
    assert not parsed.edges.duplicated().any()
    node_ids = set(parsed.nodes["hpo_id"])
    assert set(parsed.edges["child_hpo_id"]).issubset(node_ids)
    assert set(parsed.edges["parent_hpo_id"]).issubset(node_ids)


def test_unique_obsolete_replacement_is_mapped(parsed) -> None:
    assert parsed.obsolete_mapping.to_dict("records") == [
        {
            "original_hpo_id": "HP:0000100",
            "original_hpo_name": "Replaced phenotype",
            "replacement_hpo_id": "HP:0000002",
            "replacement_hpo_name": "Child phenotype",
            "mapping_method": "unique_replaced_by",
        }
    ]


def test_unresolved_obsolete_terms_are_audited(parsed) -> None:
    unresolved = parsed.unresolved_terms.set_index("hpo_id")
    assert unresolved.loc["HP:0000101", "reason"] == "obsolete_without_replacement"
    assert unresolved.loc["HP:0000102", "reason"] == "obsolete_with_ambiguous_replacement"
    assert unresolved.loc["HP:0000102", "replacement_candidates"] == [
        "HP:0000002",
        "HP:0000003",
    ]


def test_qc_counts_all_input_outcomes(parsed) -> None:
    assert parsed.qc["input_records"] == 7
    assert parsed.qc["output_records"] == 4
    assert parsed.qc["removed_records"] == 3
    assert parsed.qc["obsolete_records"] == 3
    assert parsed.qc["replaced_records"] == 1
    assert parsed.qc["unresolved_records"] == 2
    assert parsed.qc["input_is_a_edges"] == 4
    assert parsed.qc["output_is_a_edges"] == 3
    assert parsed.qc["duplicate_records"] == 1
    assert parsed.qc["non_hpo_terms_ignored"] == 1
    assert len(parsed.qc["source_sha256"]) == 64
    assert parsed.qc["ontology_data_version"] == "hp/test-fixture"
    assert parsed.qc["ontology_format_version"] == "1.2"
    assert parsed.qc["source_encoding"] == "utf-8"
    assert parsed.qc["parser"].startswith("pronto ")


def test_outputs_roundtrip_and_overwrite_is_explicit(parsed, tmp_path: Path) -> None:
    paths = HpoOutputPaths(
        nodes=tmp_path / "processed/hpo_nodes.parquet",
        edges=tmp_path / "processed/hpo_edges.parquet",
        obsolete_mapping=tmp_path / "interim/obsolete_hpo_mapping.parquet",
        unresolved_terms=tmp_path / "interim/unresolved_hpo_terms.tsv",
        qc=tmp_path / "outputs/qc/hpo_qc.json",
    )
    write_hpo_outputs(parsed, paths)

    pd.testing.assert_frame_equal(pd.read_parquet(paths.nodes), parsed.nodes)
    pd.testing.assert_frame_equal(pd.read_parquet(paths.edges), parsed.edges)
    assert len(pd.read_csv(paths.unresolved_terms, sep="\t")) == 2
    assert json.loads(paths.qc.read_text(encoding="utf-8"))["replaced_records"] == 1
    with pytest.raises(FileExistsError, match="overwrite=True"):
        write_hpo_outputs(parsed, paths)
    write_hpo_outputs(parsed, paths, overwrite=True)


def test_missing_input_fails_clearly(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="does not exist"):
        parse_hpo_ontology(tmp_path / "missing.obo")

