from pathlib import Path

import pandas as pd
import pytest

from phenotype_network_v0.data.disease_benchmark import (
    BenchmarkOutputPaths,
    build_cardiovascular_benchmark,
    find_ancestor_disease_ids,
    parse_frequency,
    write_benchmark_outputs,
)

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"


def _build(tmp_path: Path):
    nodes = pd.DataFrame({"hpo_id": ["HP:0000001", "HP:0000002", "HP:0000003"]})
    obsolete = pd.DataFrame({"original_hpo_id": ["HP:0000100"], "replacement_hpo_id": ["HP:0000003"]})
    nodes_path = tmp_path / "hpo_nodes.parquet"
    obsolete_path = tmp_path / "obsolete.parquet"
    nodes.to_parquet(nodes_path, index=False)
    obsolete.to_parquet(obsolete_path, index=False)
    return build_cardiovascular_benchmark(
        FIXTURES / "phenotype_small.hpoa",
        FIXTURES / "genes_to_disease_small.txt",
        FIXTURES / "mondo_small.obo",
        nodes_path,
        obsolete_path,
        minimum_phenotypes=3,
        minimum_genes=1,
    )


def test_parent_disease_labels_are_detected_for_exclusion() -> None:
    import pronto

    ontology = pronto.Ontology(FIXTURES / "mondo_small.obo", encoding="utf-8")
    disease_ids = {"MONDO:1000001", "MONDO:0021661", "MONDO:0005068"}
    assert find_ancestor_disease_ids(disease_ids, ontology) == {"MONDO:1000001"}


def test_frequency_parser_supports_registered_formats() -> None:
    assert parse_frequency("50%") == (0.5, "percentage", None)
    assert parse_frequency("1/4") == (0.25, "ratio", None)
    assert parse_frequency("20-40%") == (0.3, "percentage_range_midpoint", None)
    assert parse_frequency("HP:0040281") == (0.9, "very_frequent", None)
    assert parse_frequency("") == (0.5, "missing_default", None)
    assert parse_frequency("nonsense")[2] == "unparseable_nonempty_frequency"


def test_identifier_mapping_obsolete_replacement_and_filtering(tmp_path: Path) -> None:
    result = _build(tmp_path)
    assert result.samples["disease_id"].tolist() == ["MONDO:0021661"]
    assert result.samples.iloc[0]["is_core_case"]
    assert result.samples.iloc[0]["cardiovascular_family_ids"] == ["MONDO:1000001"]
    assert set(result.phenotypes["hpo_id"]) == {"HP:0000001", "HP:0000002", "HP:0000003"}
    assert "HP:0000100" in set(result.phenotypes["original_hpo_id"])
    assert len(result.phenotypes) == 3
    assert result.negative_phenotypes["hpo_id"].tolist() == ["HP:0000003"]
    assert len(result.conflicting_phenotypes) == 2
    assert set(result.conflicting_phenotypes["annotation_polarity"] ) == {"positive", "negative"}
    assert result.genes[["gene_id", "gene_symbol"]].to_dict("records") == [{"gene_id": "10", "gene_symbol": "GENEA"}]


def test_all_exclusions_are_audited(tmp_path: Path) -> None:
    result = _build(tmp_path)
    disease_reasons = set(result.unresolved_diseases["reason"])
    assert "outside_cardiovascular_scope" in disease_reasons
    assert "unmapped_to_mondo" in disease_reasons
    assert result.unresolved_genes.iloc[0]["reason"] == "invalid_ncbi_gene_id"
    assert result.unresolved_frequencies.iloc[0]["frequency_raw"] == "bad-value"
    assert result.qc["input_counts"]["hpo_annotations"] == 8
    assert result.qc["output_counts"]["diseases"] == 1


def test_clinical_registry_is_separate_from_graph_training(tmp_path: Path) -> None:
    result = _build(tmp_path)
    assert len(result.clinical_features) == 5
    assert set(result.clinical_features["feature_id"]) == {
        "CLIN:SUA", "CLIN:LDL_C", "CLIN:HS_CRP", "CLIN:IVUS_PLAQUE_BURDEN", "CLIN:IVUS_MIN_LUMEN_AREA"
    }
    assert not set(result.clinical_features["feature_id"]) & set(result.phenotypes["hpo_id"])


def test_outputs_roundtrip_and_require_explicit_overwrite(tmp_path: Path) -> None:
    result = _build(tmp_path)
    base = tmp_path / "out"
    paths = BenchmarkOutputPaths(
        base / "phenotypes.parquet", base / "genes.parquet", base / "samples.parquet",
        base / "negative.parquet", base / "conflicts.tsv", base / "clinical.parquet", base / "mapping.parquet",
        base / "diseases.tsv", base / "genes.tsv", base / "frequencies.tsv", base / "qc.json",
    )
    write_benchmark_outputs(result, paths)
    assert len(pd.read_parquet(paths.samples)) == 1
    with pytest.raises(FileExistsError, match="overwrite=True"):
        write_benchmark_outputs(result, paths)
    write_benchmark_outputs(result, paths, overwrite=True)