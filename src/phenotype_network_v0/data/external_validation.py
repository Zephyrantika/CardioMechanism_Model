"""Parsers for frozen external validation resources."""

from __future__ import annotations

import gzip
import re
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pronto


BIOGRID_COLUMNS = [
    "Entrez Gene Interactor A",
    "Entrez Gene Interactor B",
    "Official Symbol Interactor A",
    "Official Symbol Interactor B",
    "Experimental System",
    "Experimental System Type",
    "Organism ID Interactor A",
    "Organism ID Interactor B",
]
GWAS_COLUMNS = [
    "PUBMEDID",
    "DISEASE/TRAIT",
    "SNPS",
    "P-VALUE",
    "MAPPED_GENE",
    "UPSTREAM_GENE_ID",
    "DOWNSTREAM_GENE_ID",
    "SNP_GENE_IDS",
]
CARDIOVASCULAR_GTEX_TISSUES = (
    "Artery_Aorta",
    "Artery_Coronary",
    "Artery_Tibial",
    "Heart_Atrial_Appendage",
    "Heart_Left_Ventricle",
)


def parse_biogrid(
    path: Path,
    candidate_gene_ids: set[str],
    *,
    version: str,
    chunksize: int = 100_000,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Parse human physical BioGRID edges and a canonical symbol map."""
    edge_frames: list[pd.DataFrame] = []
    symbol_frames: list[pd.DataFrame] = []
    unresolved_frames: list[pd.DataFrame] = []
    counts: Counter[str] = Counter()
    for chunk in pd.read_csv(
        path,
        sep="\t",
        compression="zip",
        usecols=BIOGRID_COLUMNS,
        dtype=str,
        chunksize=chunksize,
        low_memory=False,
    ):
        counts["input_records"] += len(chunk)
        human_a = chunk["Organism ID Interactor A"].eq("9606")
        human_b = chunk["Organism ID Interactor B"].eq("9606")
        human = chunk.loc[human_a & human_b].copy()
        counts["removed_nonhuman_records"] += len(chunk) - len(human)
        for side in ("A", "B"):
            symbol_frames.append(pd.DataFrame({
                "gene_id": human[f"Entrez Gene Interactor {side}"].astype(str),
                "gene_symbol": human[f"Official Symbol Interactor {side}"].astype(str),
            }))
        physical = human.loc[
            human["Experimental System Type"].str.lower().eq("physical")
        ].copy()
        counts["removed_nonphysical_records"] += len(human) - len(physical)
        valid_a = physical["Entrez Gene Interactor A"].str.fullmatch(r"\d+")
        valid_b = physical["Entrez Gene Interactor B"].str.fullmatch(r"\d+")
        invalid = physical.loc[~(valid_a & valid_b)]
        if not invalid.empty:
            unresolved_frames.append(pd.DataFrame({
                "original_gene_a": invalid["Entrez Gene Interactor A"],
                "original_gene_b": invalid["Entrez Gene Interactor B"],
                "reason": "invalid_ncbi_gene_id",
            }))
        counts["removed_invalid_identifier_records"] += len(invalid)
        physical = physical.loc[valid_a & valid_b].copy()
        left = physical["Entrez Gene Interactor A"].astype(str)
        right = physical["Entrez Gene Interactor B"].astype(str)
        self_loop = left.eq(right)
        counts["removed_self_loops"] += int(self_loop.sum())
        physical = physical.loc[~self_loop].copy()
        left = physical["Entrez Gene Interactor A"].astype(str)
        right = physical["Entrez Gene Interactor B"].astype(str)
        in_candidates = left.isin(candidate_gene_ids) & right.isin(candidate_gene_ids)
        counts["removed_outside_candidate_universe"] += int((~in_candidates).sum())
        physical = physical.loc[in_candidates].copy()
        left = physical["Entrez Gene Interactor A"].astype(str)
        right = physical["Entrez Gene Interactor B"].astype(str)
        edge_frames.append(pd.DataFrame({
            "gene_a": np.where(left.astype(int) < right.astype(int), left, right),
            "gene_b": np.where(left.astype(int) < right.astype(int), right, left),
            "experimental_system": physical["Experimental System"].astype(str),
        }))
    raw_edges = pd.concat(edge_frames, ignore_index=True) if edge_frames else pd.DataFrame(
        columns=["gene_a", "gene_b", "experimental_system"]
    )
    edges = raw_edges.groupby(["gene_a", "gene_b"], as_index=False).agg(
        evidence_count=("experimental_system", "size"),
        evidence_types=(
            "experimental_system",
            lambda values: sorted(set(values.astype(str))),
        ),
    )
    transformed = np.log1p(edges["evidence_count"].to_numpy(dtype=float))
    maximum = float(transformed.max()) if len(transformed) else 0.0
    edges["weight"] = transformed / maximum if maximum > 0 else 0.0
    edges["source"] = "BioGRID"
    edges["source_version"] = version
    edges = edges.sort_values(["gene_a", "gene_b"], kind="stable").reset_index(drop=True)

    symbols = pd.concat(symbol_frames, ignore_index=True).drop_duplicates()
    symbols = symbols.loc[
        symbols["gene_id"].str.fullmatch(r"\d+")
        & symbols["gene_symbol"].notna()
        & symbols["gene_symbol"].ne("-")
    ].copy()
    ambiguity = symbols.groupby("gene_symbol")["gene_id"].nunique()
    ambiguous_symbols = set(ambiguity.loc[ambiguity.gt(1)].index)
    symbols["ambiguous_symbol"] = symbols["gene_symbol"].isin(ambiguous_symbols)
    symbols = symbols.sort_values(["gene_symbol", "gene_id"], kind="stable").reset_index(drop=True)
    unresolved = (
        pd.concat(unresolved_frames, ignore_index=True)
        if unresolved_frames
        else pd.DataFrame(columns=["original_gene_a", "original_gene_b", "reason"])
    )
    counts["physical_candidate_records"] = len(raw_edges)
    counts["output_unique_edges"] = len(edges)
    counts["removed_duplicate_edges"] = len(raw_edges) - len(edges)
    counts["symbol_mappings"] = len(symbols)
    counts["ambiguous_symbols"] = len(ambiguous_symbols)
    return edges, symbols, unresolved, dict(counts)


def unique_symbol_lookup(symbols: pd.DataFrame) -> dict[str, str]:
    """Return only unambiguous official-symbol to NCBI Gene mappings."""
    selected = symbols.loc[~symbols["ambiguous_symbol"]]
    return dict(zip(selected["gene_symbol"], selected["gene_id"], strict=True))


def parse_gwas_catalog(
    path: Path,
    symbol_lookup: dict[str, str],
    candidate_gene_ids: set[str],
    *,
    chunksize: int = 100_000,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int]]:
    """Map GWAS Catalog mapped-gene symbols to canonical candidate genes."""
    records: list[dict[str, str | float | None]] = []
    unresolved: Counter[tuple[str, str]] = Counter()
    input_records = 0
    mapped_records = 0
    for chunk in pd.read_csv(
        path,
        sep="\t",
        compression="zip",
        usecols=GWAS_COLUMNS,
        dtype=str,
        chunksize=chunksize,
        low_memory=False,
    ):
        input_records += len(chunk)
        chunk = chunk[GWAS_COLUMNS]
        for row in chunk.itertuples(index=False, name=None):
            values = dict(zip(GWAS_COLUMNS, row, strict=True))
            raw_symbols = values["MAPPED_GENE"]
            symbols = (
                set(re.split(r",\s*|\s+-\s+", str(raw_symbols)))
                if pd.notna(raw_symbols)
                else set()
            )
            mapped = {
                (symbol, symbol_lookup[symbol])
                for symbol in symbols
                if symbol in symbol_lookup
            }
            if not mapped:
                unresolved[(
                    str(values["DISEASE/TRAIT"]),
                    "no_unambiguous_mapped_gene_symbol",
                )] += 1
                continue
            mapped_records += 1
            original_ensembl_ids = ";".join(
                str(values[column])
                for column in (
                    "UPSTREAM_GENE_ID", "DOWNSTREAM_GENE_ID", "SNP_GENE_IDS"
                )
                if pd.notna(values[column])
            )
            for symbol, gene_id in sorted(mapped, key=lambda item: int(item[1])):
                if gene_id not in candidate_gene_ids:
                    continue
                p_value = pd.to_numeric(values["P-VALUE"], errors="coerce")
                records.append({
                    "gene_id": gene_id,
                    "gene_symbol": symbol,
                    "trait": str(values["DISEASE/TRAIT"]),
                    "pubmed_id": str(values["PUBMEDID"]),
                    "snps": str(values["SNPS"]),
                    "p_value": float(p_value) if pd.notna(p_value) else None,
                    "original_mapped_gene": str(raw_symbols),
                    "original_ensembl_gene_ids": original_ensembl_ids,
                    "mapping_method": "official_symbol_via_biogrid_5.0.260",
                    "source": "GWAS Catalog",
                })
    columns = [
        "gene_id", "gene_symbol", "trait", "pubmed_id", "snps", "p_value",
        "original_mapped_gene", "original_ensembl_gene_ids", "mapping_method", "source",
    ]
    support = pd.DataFrame.from_records(records, columns=columns).drop_duplicates()
    if not support.empty:
        support = support.sort_values(
            ["gene_id", "p_value", "trait"], kind="stable", na_position="last"
        ).reset_index(drop=True)
    audit = pd.DataFrame.from_records([
        {"original_trait": trait, "reason": reason, "record_count": count}
        for (trait, reason), count in sorted(unresolved.items())
    ])
    qc = {
        "input_records": input_records,
        "records_with_unambiguous_mapped_symbols": mapped_records,
        "records_without_unambiguous_mapped_symbols": input_records - mapped_records,
        "output_gene_support_rows": len(support),
        "supported_candidate_genes": int(support["gene_id"].nunique()) if len(support) else 0,
    }
    return support, audit, qc
def parse_gtex_expression(
    path: Path,
    symbol_lookup: dict[str, str],
    candidate_gene_ids: set[str],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int]]:
    """Map cardiovascular GTEx median TPM values to candidate NCBI Gene IDs."""
    frame = pd.read_csv(path, sep="\t", compression="gzip", skiprows=2, dtype={"Name": str})
    missing_tissues = set(CARDIOVASCULAR_GTEX_TISSUES) - set(frame.columns)
    if missing_tissues:
        raise ValueError(f"GTEx file lacks tissues: {', '.join(sorted(missing_tissues))}")
    frame["gene_symbol"] = frame["Description"].astype(str)
    frame["gene_id"] = frame["gene_symbol"].map(symbol_lookup)
    unresolved = frame.loc[frame["gene_id"].isna(), ["Name", "gene_symbol"]].copy()
    unresolved = unresolved.rename(columns={"Name": "original_ensembl_gene_id"})
    unresolved["reason"] = "unmapped_or_ambiguous_gene_symbol"
    mapped = frame.loc[frame["gene_id"].isin(candidate_gene_ids)].copy()
    expression = mapped.melt(
        id_vars=["Name", "gene_symbol", "gene_id"],
        value_vars=list(CARDIOVASCULAR_GTEX_TISSUES),
        var_name="tissue",
        value_name="median_tpm",
    ).rename(columns={"Name": "original_ensembl_gene_id"})
    expression["median_tpm"] = pd.to_numeric(expression["median_tpm"], errors="coerce")
    expression["source"] = "GTEx v10"
    expression = expression.sort_values(["gene_id", "tissue"], kind="stable").reset_index(drop=True)
    return expression, unresolved, {
        "input_genes": len(frame),
        "mapped_candidate_genes": int(mapped["gene_id"].nunique()),
        "unresolved_genes": len(unresolved),
        "output_expression_rows": len(expression),
        "tissues": len(CARDIOVASCULAR_GTEX_TISSUES),
    }


def parse_go_annotations(
    path: Path,
    symbol_lookup: dict[str, str],
    candidate_gene_ids: set[str],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int]]:
    """Map human GOA GAF records to candidate NCBI Gene IDs."""
    records: list[dict[str, str]] = []
    unresolved: Counter[str] = Counter()
    input_records = 0
    excluded_not = 0
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("!"):
                continue
            input_records += 1
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 15:
                unresolved["malformed_gaf_record"] += 1
                continue
            qualifiers = set(fields[3].split("|"))
            if "NOT" in qualifiers:
                excluded_not += 1
                continue
            symbol = fields[2]
            gene_id = symbol_lookup.get(symbol)
            if gene_id is None:
                unresolved["unmapped_or_ambiguous_gene_symbol"] += 1
                continue
            if gene_id not in candidate_gene_ids:
                unresolved["gene_outside_candidate_universe"] += 1
                continue
            records.append({
                "gene_id": gene_id,
                "gene_symbol": symbol,
                "go_id": fields[4],
                "evidence_code": fields[6],
                "aspect": fields[8],
                "original_database": fields[0],
                "original_database_id": fields[1],
                "source": "GOA human",
            })
    annotations = pd.DataFrame.from_records(records).drop_duplicates()
    if not annotations.empty:
        annotations = annotations.sort_values(
            ["gene_id", "go_id", "evidence_code"], kind="stable"
        ).reset_index(drop=True)
    audit = pd.DataFrame.from_records([
        {"reason": reason, "record_count": count}
        for reason, count in sorted(unresolved.items())
    ])
    return annotations, audit, {
        "input_records": input_records,
        "excluded_not_qualifier": excluded_not,
        "output_annotation_rows": len(annotations),
        "annotated_candidate_genes": int(annotations["gene_id"].nunique()) if len(annotations) else 0,
        **{f"removed_{reason}": count for reason, count in unresolved.items()},
    }


def parse_go_terms(path: Path, retained_go_ids: set[str]) -> pd.DataFrame:
    """Read metadata for retained non-obsolete GO terms."""
    ontology = pronto.Ontology(path)
    records = []
    for go_id in sorted(retained_go_ids):
        term = ontology.get(go_id)
        if term is None:
            continue
        records.append({
            "go_id": go_id,
            "go_name": term.name,
            "namespace": term.namespace,
            "obsolete": bool(term.obsolete),
        })
    return pd.DataFrame.from_records(records)