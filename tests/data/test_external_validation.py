import gzip
import zipfile
from pathlib import Path

import pandas as pd

from phenotype_network_v0.data.external_validation import (
    BIOGRID_COLUMNS,
    CARDIOVASCULAR_GTEX_TISSUES,
    GWAS_COLUMNS,
    parse_biogrid,
    parse_go_annotations,
    parse_gtex_expression,
    parse_gwas_catalog,
    unique_symbol_lookup,
)


def _write_zipped_tsv(path: Path, member: str, frame: pd.DataFrame) -> None:
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(member, frame.to_csv(sep="\t", index=False))


def test_biogrid_filters_and_deduplicates_candidate_physical_edges(tmp_path: Path) -> None:
    rows = [
        ["10", "20", "A", "B", "Affinity Capture-MS", "physical", "9606", "9606"],
        ["20", "10", "B", "A", "Two-hybrid", "physical", "9606", "9606"],
        ["10", "10", "A", "A", "Two-hybrid", "physical", "9606", "9606"],
        ["10", "30", "A", "C", "Two-hybrid", "physical", "9606", "9606"],
        ["10", "20", "A", "B", "Synthetic Lethality", "genetic", "9606", "9606"],
        ["10", "20", "A", "B", "Two-hybrid", "physical", "9606", "10090"],
    ]
    source = tmp_path / "biogrid.zip"
    _write_zipped_tsv(source, "biogrid.tsv", pd.DataFrame(rows, columns=BIOGRID_COLUMNS))
    edges, symbols, unresolved, qc = parse_biogrid(
        source, {"10", "20"}, version="test", chunksize=2
    )
    assert edges[["gene_a", "gene_b"]].to_dict("records") == [
        {"gene_a": "10", "gene_b": "20"}
    ]
    assert edges.iloc[0]["evidence_count"] == 2
    assert set(edges.iloc[0]["evidence_types"]) == {"Affinity Capture-MS", "Two-hybrid"}
    assert qc["removed_duplicate_edges"] == 1
    assert qc["removed_self_loops"] == 1
    assert qc["removed_outside_candidate_universe"] == 1
    assert unresolved.empty
    assert unique_symbol_lookup(symbols)["A"] == "10"


def test_external_symbol_mapping_and_audits(tmp_path: Path) -> None:
    symbols = pd.DataFrame({
        "gene_id": ["10", "20", "21"],
        "gene_symbol": ["GENE1", "AMBIG", "AMBIG"],
        "ambiguous_symbol": [False, True, True],
    })
    lookup = unique_symbol_lookup(symbols)
    assert lookup == {"GENE1": "10"}

    gwas_rows = [
        ["1", "heart trait", "rs1", "1e-8", "GENE1, AMBIG", None, None, "ENSG1"],
        ["2", "other trait", "rs2", "1e-6", "UNKNOWN", None, None, "ENSG2"],
    ]
    gwas_path = tmp_path / "gwas.zip"
    _write_zipped_tsv(gwas_path, "gwas.tsv", pd.DataFrame(gwas_rows, columns=GWAS_COLUMNS))
    gwas, gwas_audit, gwas_qc = parse_gwas_catalog(
        gwas_path, lookup, {"10"}, chunksize=1
    )
    assert gwas[["gene_id", "gene_symbol"]].to_dict("records") == [
        {"gene_id": "10", "gene_symbol": "GENE1"}
    ]
    assert gwas_qc["records_without_unambiguous_mapped_symbols"] == 1
    assert gwas_audit["record_count"].sum() == 1

    gtex_path = tmp_path / "gtex.gct.gz"
    columns = ["Name", "Description", *CARDIOVASCULAR_GTEX_TISSUES]
    gtex = pd.DataFrame([
        ["ENSG1.1", "GENE1", 1, 2, 3, 4, 5],
        ["ENSG2.1", "UNKNOWN", 0, 0, 0, 0, 0],
    ], columns=columns)
    with gzip.open(gtex_path, "wt", encoding="utf-8") as handle:
        handle.write("#1.2\n2\t5\n")
        gtex.to_csv(handle, sep="\t", index=False)
    expression, gtex_audit, gtex_qc = parse_gtex_expression(
        gtex_path, lookup, {"10"}
    )
    assert len(expression) == len(CARDIOVASCULAR_GTEX_TISSUES)
    assert gtex_qc["mapped_candidate_genes"] == 1
    assert len(gtex_audit) == 1

    goa_path = tmp_path / "goa.gaf.gz"
    base = ["UniProtKB", "P1", "GENE1", "", "GO:1", "PMID:1", "IDA", "", "P", "", "", "protein", "taxon:9606", "20260101", "GO_Central"]
    denied = base.copy()
    denied[3] = "NOT"
    with gzip.open(goa_path, "wt", encoding="utf-8") as handle:
        handle.write("!gaf-version: 2.2\n")
        handle.write("\t".join(base) + "\n")
        handle.write("\t".join(denied) + "\n")
    annotations, go_audit, go_qc = parse_go_annotations(goa_path, lookup, {"10"})
    assert len(annotations) == 1
    assert go_qc["excluded_not_qualifier"] == 1
    assert go_audit.empty