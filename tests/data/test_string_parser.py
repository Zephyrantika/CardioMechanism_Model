import json
from pathlib import Path

import pandas as pd
import pytest

from phenotype_network_v0.data.string_parser import PpiOutputPaths, clean_string_ppi, write_ppi_outputs

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"


def _clean():
    return clean_string_ppi(
        FIXTURES / "string_links_small.txt",
        FIXTURES / "string_aliases_small.txt",
        thresholds=(400, 700, 900),
        chunk_size=2,
        string_version="test",
        reference_gene_ids={"10", "20", "999"},
        core_gene_ids={"MONDO:test": {"10", "999"}},
    )


def test_identifier_mapping_and_ambiguous_alias_audit() -> None:
    result = _clean()
    assert set(result.protein_mapping["gene_id"]) == {"10", "20", "40"}
    unresolved = result.unresolved_proteins.set_index("string_protein_id")
    assert unresolved.loc["9606.P4", "reason"] == "ambiguous_ncbi_gene_mapping"
    assert unresolved.loc["9606.P5", "reason"] == "no_unique_ncbi_gene_mapping"


def test_duplicate_edges_are_undirected_and_evidence_uses_maximum() -> None:
    edge = _clean().networks[900].iloc[0]
    assert (edge["gene_a"], edge["gene_b"]) == ("10", "20")
    assert edge["combined_score"] == 900
    assert edge["neighborhood"] == 200
    assert edge["experiments"] == 500
    assert edge["database"] == 600
    assert edge["protein_pair_count"] == 2
    assert edge["weight"] == pytest.approx(0.9)


def test_thresholds_self_loops_and_dimensions() -> None:
    result = _clean()
    assert {threshold: len(frame) for threshold, frame in result.networks.items()} == {400: 2, 700: 1, 900: 1}
    for frame in result.networks.values():
        assert not (frame["gene_a"] == frame["gene_b"]).any()
        assert not frame.duplicated(["gene_a", "gene_b"]).any()
        assert frame["weight"].between(0, 1).all()
    assert result.qc["transformation_counts"]["removed_gene_self_loops"] == 1
    assert result.qc["benchmark_gene_coverage"]["fraction"] == pytest.approx(2 / 3)
    assert result.qc["core_case_gene_coverage"]["MONDO:test"]["covered_genes"] == 1
    assert result.qc["graph_summaries"]["400"]["largest_component_nodes"] == 3


def test_output_roundtrip_requires_explicit_overwrite(tmp_path: Path) -> None:
    result = _clean()
    paths = PpiOutputPaths(
        networks={value: tmp_path / f"ppi_{value}.parquet" for value in (400, 700, 900)},
        protein_mapping=tmp_path / "mapping.parquet",
        unresolved_proteins=tmp_path / "unresolved.tsv",
        qc=tmp_path / "qc.json",
    )
    write_ppi_outputs(result, paths)
    assert len(pd.read_parquet(paths.networks[400])) == 2
    assert json.loads(paths.qc.read_text())["threshold_edge_counts"]["900"] == 1
    with pytest.raises(FileExistsError, match="overwrite=True"):
        write_ppi_outputs(result, paths)
    write_ppi_outputs(result, paths, overwrite=True)