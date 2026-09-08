"""Clean Reactome v97 into gene-pathway and pathway-hierarchy tables."""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

GENE_PATHWAY_COLUMNS = ["gene_id", "pathway_id", "pathway_name", "evidence_code", "weight", "source", "source_version"]
HIERARCHY_COLUMNS = ["parent_pathway_id", "child_pathway_id", "relation_type", "source", "source_version"]
PATHWAY_COLUMNS = ["pathway_id", "pathway_name", "species", "gene_count", "included_in_primary_graph", "source_version"]
CATEGORY_RULES = {
    "immune_inflammation": r"immune|immun|interleukin|cytokine|inflamma|toll-like|complement|neutrophil|macrophage|lymphocyte",
    "lipid_metabolism": r"lipid|cholesterol|lipoprotein|fatty acid|triglyceride|phospholipid|sphingolipid",
    "purine_urate_metabolism": r"purine (?:metabolism|catabolism|salvage|ribonucleoside)|urate|uric acid|xanthine|hypoxanthine|adenosine metabolism|inosine metabolism",
    "oxidative_stress": r"oxidative stress|reactive oxygen|redox|glutathione|peroxide|nrf2",
    "cardiac_energy_metabolism": r"cardiac energy|myocardial energy|respiratory electron|oxidative phosphorylation|citric acid|tca cycle|atp synthesis|mitochondrial fatty acid beta-oxidation|pyruvate metabolism",
}


@dataclass(frozen=True)
class PathwayResult:
    """In-memory Reactome outputs and QC."""

    gene_pathway: pd.DataFrame
    hierarchy: pd.DataFrame
    pathways: pd.DataFrame
    immunometabolic_registry: pd.DataFrame
    excluded_pathways: pd.DataFrame
    unresolved_records: pd.DataFrame
    qc: dict[str, Any]


@dataclass(frozen=True)
class PathwayOutputPaths:
    """Locations of Milestone 4 artifacts."""

    gene_pathway: Path
    hierarchy: Path
    pathways: Path
    immunometabolic_registry: Path
    excluded_pathways: Path
    unresolved_records: Path
    qc: Path


def clean_reactome(
    gene_pathway_path: Path,
    hierarchy_path: Path,
    metadata_path: Path,
    *,
    species_name: str = "Homo sapiens",
    minimum_genes: int = 5,
    maximum_genes: int = 500,
    reactome_version: str = "97",
    reference_gene_ids: set[str] | None = None,
) -> PathwayResult:
    """Create canonical human Reactome tables and audit every exclusion."""
    inputs = [Path(gene_pathway_path), Path(hierarchy_path), Path(metadata_path)]
    missing = [str(path) for path in inputs if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Reactome inputs do not exist: {', '.join(missing)}")
    if minimum_genes < 1 or maximum_genes < minimum_genes:
        raise ValueError("Invalid Reactome pathway size bounds")

    metadata_raw = pd.read_csv(inputs[2], sep="\t", header=None, names=["pathway_id", "pathway_name", "species"],
                               dtype=str, keep_default_na=False)
    metadata_human = metadata_raw.loc[metadata_raw["species"].eq(species_name)].copy()
    metadata_human = metadata_human.drop_duplicates(["pathway_id", "pathway_name", "species"])
    duplicate_ids = metadata_human.groupby("pathway_id")["pathway_name"].nunique()
    ambiguous_ids = set(duplicate_ids[duplicate_ids > 1].index)
    metadata_human = metadata_human.loc[~metadata_human["pathway_id"].isin(ambiguous_ids)].drop_duplicates("pathway_id")
    names = metadata_human.set_index("pathway_id")["pathway_name"].to_dict()

    raw = pd.read_csv(inputs[0], sep="\t", header=None,
                      names=["gene_id", "pathway_id", "url", "source_pathway_name", "evidence_code", "species"],
                      dtype=str, keep_default_na=False)
    human = raw.loc[raw["species"].eq(species_name)].copy()
    valid_gene = human["gene_id"].str.fullmatch(r"\d+")
    valid_pathway = human["pathway_id"].isin(names)
    unresolved = human.loc[~valid_gene | ~valid_pathway].copy()
    unresolved["reason"] = ""
    unresolved.loc[~valid_gene, "reason"] = "invalid_ncbi_gene_id"
    unresolved.loc[valid_gene & ~valid_pathway, "reason"] = "missing_or_ambiguous_human_pathway_metadata"
    valid = human.loc[valid_gene & valid_pathway].copy()
    valid["pathway_name"] = valid["pathway_id"].map(names)
    valid = valid.drop_duplicates(["gene_id", "pathway_id", "evidence_code"])
    sizes = valid.groupby("pathway_id")["gene_id"].nunique().rename("gene_count")
    included_ids = set(sizes[(sizes >= minimum_genes) & (sizes <= maximum_genes)].index)

    pathway_table = metadata_human[["pathway_id", "pathway_name", "species"]].copy()
    pathway_table["gene_count"] = pathway_table["pathway_id"].map(sizes).fillna(0).astype(int)
    pathway_table["included_in_primary_graph"] = pathway_table["pathway_id"].isin(included_ids)
    pathway_table["source_version"] = reactome_version
    pathway_table = pathway_table.sort_values("pathway_id", kind="stable").reset_index(drop=True)
    excluded = pathway_table.loc[~pathway_table["included_in_primary_graph"]].copy()
    excluded["reason"] = excluded["gene_count"].map(
        lambda count: "below_minimum_genes" if count < minimum_genes else "above_maximum_genes"
    )
    if ambiguous_ids:
        ambiguous_rows = pd.DataFrame({
            "pathway_id": sorted(ambiguous_ids), "pathway_name": "", "species": species_name,
            "gene_count": 0, "included_in_primary_graph": False, "source_version": reactome_version,
            "reason": "ambiguous_pathway_metadata",
        })
        excluded = pd.concat([excluded, ambiguous_rows], ignore_index=True)

    gene_pathway = valid.loc[valid["pathway_id"].isin(included_ids),
                             ["gene_id", "pathway_id", "pathway_name", "evidence_code"]].copy()
    gene_pathway["weight"] = gene_pathway["pathway_id"].map(lambda value: 1.0 / math.sqrt(int(sizes[value])))
    gene_pathway["source"] = "Reactome"
    gene_pathway["source_version"] = reactome_version
    gene_pathway = gene_pathway[GENE_PATHWAY_COLUMNS].sort_values(
        ["pathway_id", "gene_id", "evidence_code"], kind="stable"
    ).reset_index(drop=True)

    hierarchy_raw = pd.read_csv(inputs[1], sep="\t", header=None,
                                names=["parent_pathway_id", "child_pathway_id"], dtype=str, keep_default_na=False)
    hierarchy = hierarchy_raw.loc[
        hierarchy_raw["parent_pathway_id"].isin(included_ids)
        & hierarchy_raw["child_pathway_id"].isin(included_ids)
        & hierarchy_raw["parent_pathway_id"].ne(hierarchy_raw["child_pathway_id"])
    ].drop_duplicates().copy()
    hierarchy["relation_type"] = "has_event"
    hierarchy["source"] = "Reactome"
    hierarchy["source_version"] = reactome_version
    hierarchy = hierarchy[HIERARCHY_COLUMNS].sort_values(
        ["parent_pathway_id", "child_pathway_id"], kind="stable"
    ).reset_index(drop=True)

    registry_records = []
    for category, pattern in CATEGORY_RULES.items():
        selected = pathway_table.loc[pathway_table["pathway_name"].str.contains(pattern, case=False, regex=True, na=False)]
        for row in selected.itertuples(index=False):
            registry_records.append({"pathway_id": row.pathway_id, "pathway_name": row.pathway_name,
                                     "category": category, "rule_pattern": pattern,
                                     "included_in_primary_graph": row.included_in_primary_graph,
                                     "source_version": reactome_version})
    registry = pd.DataFrame.from_records(registry_records).drop_duplicates(
        ["pathway_id", "category"]
    ).sort_values(["category", "pathway_id"], kind="stable").reset_index(drop=True)

    graph_genes = set(gene_pathway["gene_id"])
    reference_gene_ids = {str(value) for value in (reference_gene_ids or set())}
    qc = {
        "sources": {"gene_pathway": _source_info(inputs[0], reactome_version),
                    "hierarchy": _source_info(inputs[1], reactome_version),
                    "metadata": _source_info(inputs[2], reactome_version)},
        "species": species_name,
        "size_bounds": {"minimum": minimum_genes, "maximum": maximum_genes},
        "input_counts": {"gene_pathway_rows": len(raw), "hierarchy_rows": len(hierarchy_raw),
                         "metadata_rows": len(metadata_raw)},
        "output_counts": {"gene_pathway_rows": len(gene_pathway), "hierarchy_rows": len(hierarchy),
                          "pathways": int(pathway_table["included_in_primary_graph"].sum()),
                          "registry_rows": len(registry)},
        "removed_counts": {"nonhuman_gene_pathway_rows": len(raw) - len(human),
                           "invalid_or_unresolved_records": len(unresolved),
                           "excluded_pathways": len(excluded)},
        "unresolved_records": len(unresolved),
        "benchmark_gene_coverage": {"reference_genes": len(reference_gene_ids),
                                    "covered_genes": len(reference_gene_ids & graph_genes),
                                    "fraction": len(reference_gene_ids & graph_genes) / len(reference_gene_ids)
                                    if reference_gene_ids else None},
        "category_counts": registry.groupby("category")["pathway_id"].nunique().to_dict(),
    }
    return PathwayResult(gene_pathway, hierarchy, pathway_table, registry, excluded.reset_index(drop=True),
                         unresolved.reset_index(drop=True), qc)


def write_pathway_outputs(result: PathwayResult, paths: PathwayOutputPaths, *, overwrite: bool = False) -> None:
    """Write all pathway artifacts after checking output conflicts as a group."""
    destinations = [Path(value) for value in paths.__dict__.values()]
    existing = [path for path in destinations if path.exists()]
    if existing and not overwrite:
        raise FileExistsError("Output files already exist; use overwrite=True: " + ", ".join(map(str, existing)))
    for path in destinations:
        path.parent.mkdir(parents=True, exist_ok=True)
    result.gene_pathway.to_parquet(paths.gene_pathway, index=False)
    result.hierarchy.to_parquet(paths.hierarchy, index=False)
    result.pathways.to_parquet(paths.pathways, index=False)
    result.immunometabolic_registry.to_parquet(paths.immunometabolic_registry, index=False)
    result.excluded_pathways.to_csv(paths.excluded_pathways, sep="\t", index=False)
    result.unresolved_records.to_csv(paths.unresolved_records, sep="\t", index=False)
    paths.qc.write_text(json.dumps(result.qc, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def default_pathway_output_paths(config: dict[str, Any]) -> PathwayOutputPaths:
    """Build portable Milestone 4 output paths from configuration."""
    paths = config["paths"]
    processed, interim, outputs = Path(paths["processed"]), Path(paths["interim"]), Path(paths["outputs"])
    return PathwayOutputPaths(
        processed / "reactome_gene_pathway.parquet",
        processed / "reactome_pathway_hierarchy.parquet",
        processed / "reactome_pathways.parquet",
        processed / "immunometabolic_pathway_registry.parquet",
        interim / "excluded_reactome_pathways.tsv",
        interim / "unresolved_reactome_records.tsv",
        outputs / "qc" / "pathway_qc.json",
    )


def _source_info(path: Path, version: str) -> dict[str, Any]:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"file_name": path.name, "sha256": digest.hexdigest(), "bytes": path.stat().st_size,
            "version": version}