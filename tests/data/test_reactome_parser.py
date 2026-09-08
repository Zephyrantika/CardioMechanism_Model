import json
import math
from pathlib import Path

import pandas as pd
import pytest

from phenotype_network_v0.data.reactome_parser import PathwayOutputPaths, clean_reactome, write_pathway_outputs

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"


def _clean():
    return clean_reactome(
        FIXTURES / "reactome_gene_pathway_small.txt",
        FIXTURES / "reactome_hierarchy_small.txt",
        FIXTURES / "reactome_pathways_small.txt",
        minimum_genes=2,
        maximum_genes=3,
        reactome_version="test",
        reference_gene_ids={"10", "20", "999"},
    )


def test_human_pathways_use_metadata_names_and_size_filter() -> None:
    result = _clean()
    assert set(result.gene_pathway["pathway_id"]) == {"R-HSA-1", "R-HSA-2"}
    assert set(result.gene_pathway.loc[result.gene_pathway.pathway_id.eq("R-HSA-1"), "pathway_name"]) == {"Immune cytokine signaling"}
    assert result.excluded_pathways.set_index("pathway_id").loc["R-HSA-3", "reason"] == "above_maximum_genes"
    assert result.excluded_pathways.set_index("pathway_id").loc["R-HSA-5", "reason"] == "below_minimum_genes"


def test_gene_pathway_weights_duplicates_and_ids() -> None:
    result = _clean()
    assert len(result.gene_pathway) == 5
    p1 = result.gene_pathway[result.gene_pathway.pathway_id.eq("R-HSA-1")]
    assert p1.iloc[0]["weight"] == pytest.approx(1 / math.sqrt(2))
    assert result.gene_pathway["gene_id"].str.fullmatch(r"\d+").all()
    assert not result.gene_pathway.duplicated(["gene_id", "pathway_id", "evidence_code"]).any()


def test_hierarchy_and_immunometabolic_registry() -> None:
    result = _clean()
    assert result.hierarchy[["parent_pathway_id", "child_pathway_id"]].to_dict("records") == [
        {"parent_pathway_id": "R-HSA-1", "child_pathway_id": "R-HSA-2"}
    ]
    categories = set(result.immunometabolic_registry["category"])
    assert {"immune_inflammation", "purine_urate_metabolism", "lipid_metabolism"} <= categories


def test_unresolved_and_coverage_are_reported() -> None:
    result = _clean()
    assert set(result.unresolved_records["reason"]) == {
        "invalid_ncbi_gene_id", "missing_or_ambiguous_human_pathway_metadata"
    }
    assert result.qc["benchmark_gene_coverage"]["fraction"] == pytest.approx(2 / 3)


def test_output_roundtrip_and_overwrite(tmp_path: Path) -> None:
    result = _clean()
    paths = PathwayOutputPaths(
        tmp_path / "gene.parquet", tmp_path / "hierarchy.parquet", tmp_path / "pathways.parquet",
        tmp_path / "registry.parquet", tmp_path / "excluded.tsv", tmp_path / "unresolved.tsv", tmp_path / "qc.json",
    )
    write_pathway_outputs(result, paths)
    assert len(pd.read_parquet(paths.gene_pathway)) == 5
    assert json.loads(paths.qc.read_text())["output_counts"]["pathways"] == 2
    with pytest.raises(FileExistsError, match="overwrite=True"):
        write_pathway_outputs(result, paths)
    write_pathway_outputs(result, paths, overwrite=True)