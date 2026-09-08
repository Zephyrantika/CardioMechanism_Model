"""Build the auditable cardiovascular disease benchmark for Milestone 2."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import pronto

CARDIOVASCULAR_ROOT = "MONDO:0004995"
CORE_DISEASES = {
    "MONDO:0021661": "coronary atherosclerosis",
    "MONDO:0005068": "myocardial infarction",
    "MONDO:0005252": "heart failure",
}
FREQUENCY_TERMS = {
    "HP:0040280": (1.00, "obligate"),
    "HP:0040281": (0.90, "very_frequent"),
    "HP:0040282": (0.70, "frequent"),
    "HP:0040283": (0.40, "occasional"),
    "HP:0040284": (0.20, "very_rare"),
}

PHENOTYPE_COLUMNS = [
    "disease_id", "disease_name", "original_disease_id", "original_disease_name",
    "hpo_id", "original_hpo_id", "frequency_weight", "frequency_raw",
    "frequency_method", "reference", "evidence", "onset", "sex", "modifier",
    "aspect", "biocuration", "source_database", "mapping_method",
]
GENE_COLUMNS = [
    "disease_id", "disease_name", "original_disease_id", "gene_id", "gene_symbol",
    "original_gene_id", "association_type", "source", "mapping_method",
]
SAMPLE_COLUMNS = [
    "disease_id", "disease_name", "positive_phenotype_count", "negative_phenotype_count",
    "known_gene_count", "cardiovascular_family_ids", "is_core_case",
    "cardiovascular_inclusion_reason",
]
MAPPING_COLUMNS = [
    "original_disease_id", "original_disease_name", "canonical_disease_id",
    "canonical_disease_name", "mapping_method", "mapping_candidates",
    "is_cardiovascular", "cardiovascular_family_ids", "is_core_case", "reason",
]
UNRESOLVED_DISEASE_COLUMNS = [
    "original_disease_id", "original_disease_name", "source_table", "reason",
    "mapping_candidates",
]
UNRESOLVED_GENE_COLUMNS = [
    "original_gene_id", "gene_symbol", "original_disease_id", "reason", "source",
]
UNRESOLVED_FREQUENCY_COLUMNS = [
    "original_disease_id", "original_hpo_id", "frequency_raw", "reason",
]


@dataclass(frozen=True)
class BenchmarkResult:
    """In-memory Milestone 2 tables and QC summary."""

    phenotypes: pd.DataFrame
    genes: pd.DataFrame
    samples: pd.DataFrame
    negative_phenotypes: pd.DataFrame
    conflicting_phenotypes: pd.DataFrame
    clinical_features: pd.DataFrame
    disease_mapping: pd.DataFrame
    unresolved_diseases: pd.DataFrame
    unresolved_genes: pd.DataFrame
    unresolved_frequencies: pd.DataFrame
    qc: dict[str, Any]


@dataclass(frozen=True)
class BenchmarkOutputPaths:
    """Locations of all Milestone 2 output artifacts."""

    phenotypes: Path
    genes: Path
    samples: Path
    negative_phenotypes: Path
    conflicting_phenotypes: Path
    clinical_features: Path
    disease_mapping: Path
    unresolved_diseases: Path
    unresolved_genes: Path
    unresolved_frequencies: Path
    qc: Path


def parse_frequency(raw_value: str) -> tuple[float, str, str | None]:
    """Convert an HPO frequency value to a deterministic weight and audit reason."""
    raw = str(raw_value or "").strip()
    if not raw:
        return 0.5, "missing_default", None
    if raw in FREQUENCY_TERMS:
        weight, method = FREQUENCY_TERMS[raw]
        return weight, method, None
    percent = re.fullmatch(r"(\d+(?:\.\d+)?)\s*%", raw)
    if percent:
        value = float(percent.group(1)) / 100.0
        if 0.0 <= value <= 1.0:
            return value, "percentage", None
    ratio = re.fullmatch(r"(\d+(?:\.\d+)?)\s*/\s*(\d+(?:\.\d+)?)", raw)
    if ratio and float(ratio.group(2)) > 0:
        value = float(ratio.group(1)) / float(ratio.group(2))
        if 0.0 <= value <= 1.0:
            return value, "ratio", None
    interval = re.fullmatch(r"(\d+(?:\.\d+)?)\s*[-–]\s*(\d+(?:\.\d+)?)\s*%", raw)
    if interval:
        low, high = float(interval.group(1)), float(interval.group(2))
        if 0.0 <= low <= high <= 100.0:
            return ((low + high) / 2.0) / 100.0, "percentage_range_midpoint", None
    return 0.5, "unparsed_default", "unparseable_nonempty_frequency"


def build_cardiovascular_benchmark(
    hpoa_path: Path,
    gene_disease_path: Path,
    mondo_path: Path,
    hpo_nodes_path: Path,
    obsolete_hpo_mapping_path: Path,
    *,
    minimum_phenotypes: int = 3,
    minimum_genes: int = 1,
    cardiovascular_root: str = CARDIOVASCULAR_ROOT,
) -> BenchmarkResult:
    """Build a disease-level cardiovascular benchmark from versioned public sources."""
    inputs = [hpoa_path, gene_disease_path, mondo_path, hpo_nodes_path, obsolete_hpo_mapping_path]
    missing = [str(Path(path)) for path in inputs if not Path(path).is_file()]
    if missing:
        raise FileNotFoundError(f"Required benchmark inputs do not exist: {', '.join(missing)}")
    if minimum_phenotypes < 1 or minimum_genes < 1:
        raise ValueError("Benchmark minimum counts must be positive")

    hpoa, hpo_metadata = _read_hpoa(Path(hpoa_path))
    gene_source = _read_tsv(Path(gene_disease_path), {
        "ncbi_gene_id", "gene_symbol", "association_type", "disease_id", "source"
    })
    hpo_nodes = pd.read_parquet(hpo_nodes_path)
    obsolete = pd.read_parquet(obsolete_hpo_mapping_path)
    active_hpo = set(hpo_nodes["hpo_id"].astype(str))
    obsolete_lookup = dict(zip(obsolete["original_hpo_id"], obsolete["replacement_hpo_id"]))

    ontology = pronto.Ontology(Path(mondo_path), encoding="utf-8")
    mondo = _build_mondo_index(ontology, cardiovascular_root)
    disease_sources = pd.concat([
        hpoa[["database_id", "disease_name"]].rename(columns={"database_id": "disease_id"}),
        gene_source[["disease_id"]].assign(disease_name=""),
    ], ignore_index=True).drop_duplicates()
    disease_sources = disease_sources.groupby("disease_id", as_index=False, sort=True).agg(
        disease_name=("disease_name", lambda values: next((value for value in values if value), ""))
    )
    mapping = _map_diseases(disease_sources, mondo)
    map_lookup = mapping.drop_duplicates("original_disease_id").set_index("original_disease_id")

    phenotype_records: list[dict[str, Any]] = []
    cardiovascular_hpoa_rows = 0
    unresolved_frequency_records: list[dict[str, str]] = []
    invalid_hpo_records: list[dict[str, str]] = []
    for row in hpoa.itertuples(index=False):
        disease = map_lookup.loc[row.database_id]
        if not bool(disease["is_cardiovascular"]):
            continue
        cardiovascular_hpoa_rows += 1
        original_hpo = str(row.hpo_id).strip()
        hpo_id = obsolete_lookup.get(original_hpo, original_hpo)
        if hpo_id not in active_hpo:
            invalid_hpo_records.append({
                "original_disease_id": row.database_id,
                "original_hpo_id": original_hpo,
                "frequency_raw": row.frequency,
                "reason": "unresolved_hpo_id",
            })
            continue
        weight, frequency_method, frequency_reason = parse_frequency(row.frequency)
        if frequency_reason:
            unresolved_frequency_records.append({
                "original_disease_id": row.database_id,
                "original_hpo_id": original_hpo,
                "frequency_raw": row.frequency,
                "reason": frequency_reason,
            })
        phenotype_records.append({
            "disease_id": disease["canonical_disease_id"],
            "disease_name": disease["canonical_disease_name"],
            "original_disease_id": row.database_id,
            "original_disease_name": row.disease_name,
            "hpo_id": hpo_id,
            "original_hpo_id": original_hpo,
            "frequency_weight": weight,
            "frequency_raw": row.frequency,
            "frequency_method": frequency_method,
            "reference": row.reference,
            "evidence": row.evidence,
            "onset": row.onset,
            "sex": row.sex,
            "modifier": row.modifier,
            "aspect": row.aspect,
            "biocuration": row.biocuration,
            "source_database": row.database_id.split(":", 1)[0],
            "mapping_method": disease["mapping_method"],
            "is_negative": str(row.qualifier).strip().upper() == "NOT",
        })

    phenotype_frame = pd.DataFrame.from_records(phenotype_records)
    if phenotype_frame.empty:
        phenotype_frame = pd.DataFrame(columns=[*PHENOTYPE_COLUMNS, "is_negative"])
    positive = phenotype_frame.loc[~phenotype_frame["is_negative"]].drop(columns="is_negative")
    negative = phenotype_frame.loc[phenotype_frame["is_negative"]].drop(columns="is_negative")
    positive = _deduplicate(positive, PHENOTYPE_COLUMNS, ["disease_id", "hpo_id", "reference", "evidence"])
    negative = _deduplicate(negative, PHENOTYPE_COLUMNS, ["disease_id", "hpo_id", "reference", "evidence"])

    gene_records: list[dict[str, str]] = []
    cardiovascular_gene_rows = 0
    unresolved_gene_records: list[dict[str, str]] = []
    for row in gene_source.itertuples(index=False):
        disease = map_lookup.loc[row.disease_id]
        original_gene = str(row.ncbi_gene_id).strip()
        if not bool(disease["is_cardiovascular"]):
            continue
        cardiovascular_gene_rows += 1
        match = re.fullmatch(r"(?:NCBIGene:)?(\d+)", original_gene, flags=re.IGNORECASE)
        if not match:
            unresolved_gene_records.append({
                "original_gene_id": original_gene, "gene_symbol": row.gene_symbol,
                "original_disease_id": row.disease_id, "reason": "invalid_ncbi_gene_id",
                "source": row.source,
            })
            continue
        gene_records.append({
            "disease_id": disease["canonical_disease_id"],
            "disease_name": disease["canonical_disease_name"],
            "original_disease_id": row.disease_id,
            "gene_id": match.group(1),
            "gene_symbol": row.gene_symbol,
            "original_gene_id": original_gene,
            "association_type": row.association_type,
            "source": row.source,
            "mapping_method": disease["mapping_method"],
        })
    genes = _deduplicate(pd.DataFrame.from_records(gene_records, columns=GENE_COLUMNS), GENE_COLUMNS,
                         ["disease_id", "gene_id", "association_type", "source"])

    positive_counts = positive.groupby("disease_id")["hpo_id"].nunique()
    negative_counts = negative.groupby("disease_id")["hpo_id"].nunique()
    gene_counts = genes.groupby("disease_id")["gene_id"].nunique()
    threshold_eligible = set(positive_counts[positive_counts >= minimum_phenotypes].index) & set(
        gene_counts[gene_counts >= minimum_genes].index
    )
    ancestor_diseases = find_ancestor_disease_ids(threshold_eligible, ontology)
    eligible = threshold_eligible - ancestor_diseases
    positive = positive.loc[positive["disease_id"].isin(eligible)].reset_index(drop=True)
    negative = negative.loc[negative["disease_id"].isin(eligible)].reset_index(drop=True)
    genes = genes.loc[genes["disease_id"].isin(eligible)].reset_index(drop=True)
    conflict_keys = positive[["disease_id", "hpo_id"]].drop_duplicates().merge(
        negative[["disease_id", "hpo_id"]].drop_duplicates(), on=["disease_id", "hpo_id"]
    )
    positive_conflicts = positive.merge(conflict_keys, on=["disease_id", "hpo_id"])
    positive_conflicts["annotation_polarity"] = "positive"
    negative_conflicts = negative.merge(conflict_keys, on=["disease_id", "hpo_id"])
    negative_conflicts["annotation_polarity"] = "negative"
    conflicting_phenotypes = pd.concat([positive_conflicts, negative_conflicts], ignore_index=True).sort_values(
        ["disease_id", "hpo_id", "annotation_polarity", "reference"], kind="stable"
    ).reset_index(drop=True)

    mapped_by_id = mapping.drop_duplicates("canonical_disease_id").set_index("canonical_disease_id")
    sample_records = []
    for disease_id in sorted(eligible):
        disease = mapped_by_id.loc[disease_id]
        sample_records.append({
            "disease_id": disease_id,
            "disease_name": disease["canonical_disease_name"],
            "positive_phenotype_count": int(positive_counts.get(disease_id, 0)),
            "negative_phenotype_count": int(negative_counts.get(disease_id, 0)),
            "known_gene_count": int(gene_counts.get(disease_id, 0)),
            "cardiovascular_family_ids": disease["cardiovascular_family_ids"],
            "is_core_case": bool(disease["is_core_case"]),
            "cardiovascular_inclusion_reason": "descendant_of_registered_root",
        })
    samples = pd.DataFrame.from_records(sample_records, columns=SAMPLE_COLUMNS)

    unresolved_diseases = mapping.loc[~mapping["is_cardiovascular"], [
        "original_disease_id", "original_disease_name", "reason", "mapping_candidates"
    ]].copy()
    unresolved_diseases["source_table"] = "hpoa_or_genes_to_disease"
    unresolved_diseases = unresolved_diseases[UNRESOLVED_DISEASE_COLUMNS]
    unresolved_frequencies = pd.DataFrame.from_records(
        [*unresolved_frequency_records, *invalid_hpo_records], columns=UNRESOLVED_FREQUENCY_COLUMNS
    ).drop_duplicates().reset_index(drop=True)
    unresolved_genes = pd.DataFrame.from_records(unresolved_gene_records, columns=UNRESOLVED_GENE_COLUMNS)

    qc = {
        "sources": {
            "hpoa": _source_info(Path(hpoa_path), hpo_metadata.get("version")),
            "genes_to_disease": _source_info(Path(gene_disease_path), hpo_metadata.get("version")),
            "mondo": _source_info(Path(mondo_path), str(ontology.metadata.data_version or "")),
            "hpo_nodes": _source_info(Path(hpo_nodes_path), None),
            "obsolete_hpo_mapping": _source_info(Path(obsolete_hpo_mapping_path), None),
        },
        "input_counts": {"hpo_annotations": len(hpoa), "gene_associations": len(gene_source),
                         "unique_source_diseases": int(disease_sources["disease_id"].nunique())},
        "output_counts": {"diseases": len(samples), "positive_phenotypes": len(positive),
                          "negative_phenotypes": len(negative), "disease_gene_edges": len(genes),
                          "phenotype_conflict_pairs": len(conflict_keys),
                          "clinical_features": len(clinical_feature_registry())},
        "transformation_counts": {
            "hpoa": {
                "input": len(hpoa),
                "outside_cardiovascular_scope": len(hpoa) - cardiovascular_hpoa_rows,
                "cardiovascular_rows": cardiovascular_hpoa_rows,
                "invalid_or_unresolved_hpo": len(invalid_hpo_records),
                "below_disease_threshold": len(phenotype_frame) - len(positive) - len(negative),
                "output": len(positive) + len(negative),
            },
            "genes_to_disease": {
                "input": len(gene_source),
                "outside_cardiovascular_scope": len(gene_source) - cardiovascular_gene_rows,
                "cardiovascular_rows": cardiovascular_gene_rows,
                "invalid_gene_id": len(unresolved_gene_records),
                "duplicate_or_below_disease_threshold": cardiovascular_gene_rows - len(unresolved_gene_records) - len(genes),
                "output": len(genes),
            },
            "diseases": {
                "input_unique_source_ids": int(disease_sources["disease_id"].nunique()),
                "mapped_cardiovascular": int(mapping.loc[mapping["is_cardiovascular"], "canonical_disease_id"].nunique()),
                "below_threshold": int(mapping.loc[
                    mapping["is_cardiovascular"] & ~mapping["canonical_disease_id"].isin(threshold_eligible),
                    "canonical_disease_id"].nunique()),
                "ancestor_labels_excluded": len(ancestor_diseases),
                "output": len(samples),
            },
        },
        "removed_counts": {
            "cardiovascular_diseases_below_threshold": int(mapping.loc[
                mapping["is_cardiovascular"] & ~mapping["canonical_disease_id"].isin(threshold_eligible),
                "canonical_disease_id"].nunique()),
            "ancestor_disease_labels_excluded": len(ancestor_diseases),
            "phenotype_rows_outside_final_benchmark": int(len(hpoa) - len(positive) - len(negative)),
            "gene_rows_outside_final_benchmark": int(len(gene_source) - len(genes)),
        },
        "unresolved_counts": {"diseases": len(unresolved_diseases), "genes": len(unresolved_genes),
                              "frequencies_or_hpo_ids": len(unresolved_frequencies)},
        "cardiovascular_root": cardiovascular_root,
        "core_case_coverage": {
            disease_id: {
                "has_public_phenotypes": disease_id in set(positive_counts.index),
                "has_public_genes": disease_id in set(gene_counts.index),
                "eligible": disease_id in eligible,
            }
            for disease_id in CORE_DISEASES
        },
        "minimum_phenotypes": minimum_phenotypes,
        "minimum_genes": minimum_genes,
    }
    return BenchmarkResult(positive, genes, samples, negative, conflicting_phenotypes,
                           clinical_feature_registry(), mapping, unresolved_diseases,
                           unresolved_genes, unresolved_frequencies, qc)


def find_ancestor_disease_ids(disease_ids: set[str], ontology: pronto.Ontology) -> set[str]:
    """Find benchmark labels that are ancestors of another benchmark disease."""
    ancestors: set[str] = set()
    for disease_id in sorted(disease_ids):
        if disease_id not in ontology:
            continue
        ancestors.update(
            term.id for term in ontology[disease_id].superclasses(with_self=False) if term.id in disease_ids
        )
    return ancestors


def clinical_feature_registry() -> pd.DataFrame:
    """Return the pre-registered clinical bridge without patient-level values."""
    rows = [
        ("CLIN:SUA", "serum uric acid", "continuous", "LOINC", "3084-1", "mg/dL", "nonnegative", "clinical_bridge", "LOINC"),
        ("CLIN:LDL_C", "LDL cholesterol", "continuous", "LOCAL", "CLIN:LDL_C", "mg/dL", "nonnegative", "clinical_bridge", "method-dependent assay"),
        ("CLIN:HS_CRP", "high-sensitivity C-reactive protein", "continuous", "LOINC", "30522-7", "mg/L", "nonnegative", "clinical_bridge", "LOINC"),
        ("CLIN:IVUS_PLAQUE_BURDEN", "IVUS plaque burden", "continuous", "LOCAL", "CLIN:IVUS_PLAQUE_BURDEN", "%", "0_to_100", "future_external_validation", "study protocol"),
        ("CLIN:IVUS_MIN_LUMEN_AREA", "IVUS minimum lumen area", "continuous", "LOCAL", "CLIN:IVUS_MIN_LUMEN_AREA", "mm^2", "nonnegative", "future_external_validation", "study protocol"),
    ]
    return pd.DataFrame(rows, columns=["feature_id", "feature_name", "feature_type", "canonical_system",
                                       "canonical_id", "unit", "value_domain", "role", "source"])


def write_benchmark_outputs(result: BenchmarkResult, paths: BenchmarkOutputPaths, *, overwrite: bool = False) -> None:
    """Write all benchmark artifacts after checking output conflicts as a group."""
    destinations = [Path(value) for value in paths.__dict__.values()]
    existing = [path for path in destinations if path.exists()]
    if existing and not overwrite:
        raise FileExistsError("Output files already exist; use overwrite=True: " + ", ".join(map(str, existing)))
    for path in destinations:
        path.parent.mkdir(parents=True, exist_ok=True)
    result.phenotypes.to_parquet(paths.phenotypes, index=False)
    result.genes.to_parquet(paths.genes, index=False)
    result.samples.to_parquet(paths.samples, index=False)
    result.negative_phenotypes.to_parquet(paths.negative_phenotypes, index=False)
    _write_tsv(result.conflicting_phenotypes, paths.conflicting_phenotypes)
    result.clinical_features.to_parquet(paths.clinical_features, index=False)
    result.disease_mapping.to_parquet(paths.disease_mapping, index=False)
    _write_tsv(result.unresolved_diseases, paths.unresolved_diseases)
    _write_tsv(result.unresolved_genes, paths.unresolved_genes)
    _write_tsv(result.unresolved_frequencies, paths.unresolved_frequencies)
    Path(paths.qc).write_text(json.dumps(result.qc, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def default_benchmark_output_paths(config: dict[str, Any]) -> BenchmarkOutputPaths:
    """Build portable Milestone 2 output paths from configuration."""
    paths = config["paths"]
    processed, interim, outputs = Path(paths["processed"]), Path(paths["interim"]), Path(paths["outputs"])
    return BenchmarkOutputPaths(
        processed / "cardiovascular_disease_phenotypes.parquet",
        processed / "cardiovascular_disease_genes.parquet",
        processed / "cardiovascular_disease_samples.parquet",
        processed / "cardiovascular_negative_phenotypes.parquet",
        interim / "conflicting_phenotype_annotations.tsv",
        processed / "clinical_feature_registry.parquet",
        interim / "cardiovascular_disease_mapping.parquet",
        interim / "unresolved_cardiovascular_diseases.tsv",
        interim / "unresolved_genes.tsv",
        interim / "unresolved_frequencies.tsv",
        outputs / "qc" / "cardiovascular_benchmark_qc.json",
    )


def _build_mondo_index(ontology: pronto.Ontology, root_id: str) -> dict[str, Any]:
    if root_id not in ontology:
        raise ValueError(f"Cardiovascular MONDO root is absent: {root_id}")
    active = {term.id: term for term in ontology.terms() if term.id.startswith("MONDO:") and not term.obsolete}
    cardiovascular = {term.id for term in ontology[root_id].subclasses(with_self=True) if not term.obsolete}
    direct_families = {term.id for term in ontology[root_id].subclasses(distance=1, with_self=False) if not term.obsolete}
    aliases: dict[str, set[str]] = {}
    methods: dict[tuple[str, str], str] = {}
    for term in active.values():
        for alias, method in [(term.id, "direct_mondo"), *[(str(item), "alternate_id") for item in term.alternate_ids],
                              *[(xref.id, "explicit_xref") for xref in term.xrefs]]:
            normalized = _normalize_identifier(alias)
            aliases.setdefault(normalized, set()).add(term.id)
            methods[(normalized, term.id)] = method
    families: dict[str, list[str]] = {}
    for disease_id in cardiovascular:
        if disease_id == root_id:
            families[disease_id] = [root_id]
        else:
            ancestors = {term.id for term in active[disease_id].superclasses(with_self=True)}
            families[disease_id] = sorted(ancestors & direct_families)
    return {"active": active, "cardiovascular": cardiovascular, "aliases": aliases,
            "methods": methods, "families": families}


def _map_diseases(source: pd.DataFrame, mondo: dict[str, Any]) -> pd.DataFrame:
    records = []
    for row in source.itertuples(index=False):
        original = str(row.disease_id).strip()
        normalized = _normalize_identifier(original)
        candidates = sorted(mondo["aliases"].get(normalized, set()))
        canonical = candidates[0] if len(candidates) == 1 else None
        is_cardio = canonical in mondo["cardiovascular"] if canonical else False
        if not candidates:
            reason = "unmapped_to_mondo"
        elif len(candidates) > 1:
            reason = "ambiguous_mondo_mapping"
        elif not is_cardio:
            reason = "outside_cardiovascular_scope"
        else:
            reason = ""
        records.append({
            "original_disease_id": original,
            "original_disease_name": str(row.disease_name or ""),
            "canonical_disease_id": canonical,
            "canonical_disease_name": mondo["active"][canonical].name if canonical else None,
            "mapping_method": mondo["methods"].get((normalized, canonical), "unresolved") if canonical else "unresolved",
            "mapping_candidates": candidates,
            "is_cardiovascular": is_cardio,
            "cardiovascular_family_ids": mondo["families"].get(canonical, []),
            "is_core_case": canonical in CORE_DISEASES if canonical else False,
            "reason": reason,
        })
    frame = pd.DataFrame.from_records(records, columns=MAPPING_COLUMNS)
    return frame.sort_values(["original_disease_id", "original_disease_name"], kind="stable").reset_index(drop=True)


def _normalize_identifier(value: str) -> str:
    normalized = str(value).strip()
    if normalized.upper().startswith("ORPHA:"):
        normalized = "ORPHANET:" + normalized.split(":", 1)[1]
    return normalized.upper()


def _read_hpoa(path: Path) -> tuple[pd.DataFrame, dict[str, str]]:
    metadata: dict[str, str] = {}
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            if not line.startswith("#"):
                break
            key, separator, value = line[1:].partition(":")
            if separator:
                metadata[key.strip()] = value.strip().strip('"')
    required = {"database_id", "disease_name", "qualifier", "hpo_id", "reference", "evidence",
                "onset", "frequency", "sex", "modifier", "aspect", "biocuration"}
    return _read_tsv(path, required, comment="#"), metadata


def _read_tsv(path: Path, required: set[str], **kwargs: Any) -> pd.DataFrame:
    frame = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False, **kwargs)
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Missing columns in {path}: {', '.join(missing)}")
    return frame


def _deduplicate(frame: pd.DataFrame, columns: list[str], keys: list[str]) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=columns)
    return frame[columns].drop_duplicates(subset=keys).sort_values(keys, kind="stable").reset_index(drop=True)


def _write_tsv(frame: pd.DataFrame, path: Path) -> None:
    serializable = frame.copy()
    for column in serializable.columns:
        if serializable[column].map(lambda value: isinstance(value, list)).any():
            serializable[column] = serializable[column].map(json.dumps)
    serializable.to_csv(path, sep="\t", index=False)


def _source_info(path: Path, version: str | None) -> dict[str, Any]:
    return {"file_name": path.name, "sha256": _sha256(path), "bytes": path.stat().st_size, "version": version}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()