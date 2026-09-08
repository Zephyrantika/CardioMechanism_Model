"""Clean STRING protein links into auditable gene-level PPI networks."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

TRUSTED_NCBI_SOURCE = "Ensembl_HGNC_entrez_id"
BASE_COLUMNS = ["gene_a", "gene_b", "example_protein_a", "example_protein_b", "protein_pair_count"]


@dataclass(frozen=True)
class PpiResult:
    """Cleaned threshold networks, mappings, audit rows, and QC."""

    networks: dict[int, pd.DataFrame]
    protein_mapping: pd.DataFrame
    unresolved_proteins: pd.DataFrame
    qc: dict[str, Any]


@dataclass(frozen=True)
class PpiOutputPaths:
    """Locations of Milestone 3 outputs."""

    networks: dict[int, Path]
    protein_mapping: Path
    unresolved_proteins: Path
    qc: Path


def clean_string_ppi(
    links_path: Path,
    aliases_path: Path,
    *,
    taxonomy_id: int = 9606,
    thresholds: tuple[int, ...] = (400, 700, 900),
    chunk_size: int = 500_000,
    string_version: str = "12.0",
    primary_threshold: int = 700,
    reference_gene_ids: set[str] | None = None,
    core_gene_ids: dict[str, set[str]] | None = None,
) -> PpiResult:
    """Map STRING proteins to NCBI genes and aggregate duplicate undirected PPIs."""
    links_path, aliases_path = Path(links_path), Path(aliases_path)
    for path in (links_path, aliases_path):
        if not path.is_file():
            raise FileNotFoundError(f"STRING input does not exist: {path}")
    if not thresholds or any(score < 0 or score > 1000 for score in thresholds):
        raise ValueError("STRING thresholds must be between 0 and 1000")
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    thresholds = tuple(sorted(set(int(value) for value in thresholds)))
    if primary_threshold not in thresholds:
        raise ValueError("primary_threshold must be included in thresholds")
    mapping, ambiguous = _read_alias_mapping(aliases_path, chunk_size)
    protein_to_gene = dict(zip(mapping["string_protein_id"], mapping["gene_id"]))

    chunks: list[pd.DataFrame] = []
    unresolved_counts: dict[str, int] = {}
    input_edges = species_edges = minimum_threshold_edges = mapped_edges = self_loops = 0
    score_columns: list[str] | None = None
    prefix = f"{taxonomy_id}."
    for chunk in pd.read_csv(links_path, sep=r"\s+", dtype={"protein1": str, "protein2": str}, chunksize=chunk_size):
        required = {"protein1", "protein2", "combined_score"}
        missing = required - set(chunk.columns)
        if missing:
            raise ValueError(f"Missing STRING link columns: {', '.join(sorted(missing))}")
        if score_columns is None:
            score_columns = [column for column in chunk.columns if column not in {"protein1", "protein2"}]
        input_edges += len(chunk)
        species_mask = chunk["protein1"].str.startswith(prefix) & chunk["protein2"].str.startswith(prefix)
        species_edges += int(species_mask.sum())
        chunk = chunk.loc[species_mask & (chunk["combined_score"] >= thresholds[0])].copy()
        minimum_threshold_edges += len(chunk)
        chunk["gene1"] = chunk["protein1"].map(protein_to_gene)
        chunk["gene2"] = chunk["protein2"].map(protein_to_gene)
        missing_mask = chunk["gene1"].isna() | chunk["gene2"].isna()
        for column in ("protein1", "protein2"):
            is_unmapped = chunk[column].map(lambda value: value not in protein_to_gene).astype(bool)
            values = chunk.loc[missing_mask & is_unmapped, column]
            for protein_id, count in values.value_counts().items():
                unresolved_counts[protein_id] = unresolved_counts.get(protein_id, 0) + int(count)
        chunk = chunk.loc[~missing_mask].copy()
        mapped_edges += len(chunk)
        self_mask = chunk["gene1"] == chunk["gene2"]
        self_loops += int(self_mask.sum())
        chunk = chunk.loc[~self_mask]
        chunk["gene_a"] = chunk[["gene1", "gene2"]].min(axis=1).astype(str)
        chunk["gene_b"] = chunk[["gene1", "gene2"]].max(axis=1).astype(str)
        swap = chunk["gene1"] != chunk["gene_a"]
        chunk["example_protein_a"] = chunk["protein1"].where(~swap, chunk["protein2"])
        chunk["example_protein_b"] = chunk["protein2"].where(~swap, chunk["protein1"])
        chunk["protein_pair_count"] = 1
        aggregations: dict[str, str] = {column: "max" for column in score_columns}
        aggregations.update({"example_protein_a": "min", "example_protein_b": "min", "protein_pair_count": "sum"})
        grouped = chunk.groupby(["gene_a", "gene_b"], as_index=False, sort=False).agg(aggregations)
        chunks.append(grouped)
    if score_columns is None:
        raise ValueError("STRING links file contains no records")

    combined = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame(columns=[*BASE_COLUMNS, *score_columns])
    aggregations = {column: "max" for column in score_columns}
    aggregations.update({"example_protein_a": "min", "example_protein_b": "min", "protein_pair_count": "sum"})
    combined = combined.groupby(["gene_a", "gene_b"], as_index=False, sort=True).agg(aggregations)
    combined["weight"] = combined["combined_score"].astype(float) / 1000.0
    combined["source"] = "STRING"
    combined["source_version"] = string_version
    ordered = ["gene_a", "gene_b", *score_columns, "weight", "example_protein_a", "example_protein_b",
               "protein_pair_count", "source", "source_version"]
    networks = {
        threshold: combined.loc[combined["combined_score"] >= threshold, ordered].reset_index(drop=True)
        for threshold in thresholds
    }

    unresolved = pd.DataFrame(
        [{"string_protein_id": protein, "edge_occurrences": count, "reason": "no_unique_ncbi_gene_mapping"}
         for protein, count in sorted(unresolved_counts.items())],
        columns=["string_protein_id", "edge_occurrences", "reason"],
    )
    ambiguous_audit = ambiguous.assign(edge_occurrences=0, reason="ambiguous_ncbi_gene_mapping")
    unresolved = pd.concat([ambiguous_audit[["string_protein_id", "edge_occurrences", "reason"]], unresolved],
                           ignore_index=True).drop_duplicates("string_protein_id").sort_values("string_protein_id")
    graph_summaries = {str(threshold): _graph_summary(frame) for threshold, frame in networks.items()}
    primary_genes = set(networks[primary_threshold]["gene_a"]) | set(networks[primary_threshold]["gene_b"])
    reference_gene_ids = {str(value) for value in (reference_gene_ids or set())}
    core_gene_ids = {key: {str(value) for value in values} for key, values in (core_gene_ids or {}).items()}
    qc = {
        "sources": {
            "links": _source_info(links_path, string_version),
            "aliases": _source_info(aliases_path, string_version),
        },
        "taxonomy_id": taxonomy_id,
        "trusted_mapping_source": TRUSTED_NCBI_SOURCE,
        "input_edges": input_edges,
        "transformation_counts": {
            "outside_species": input_edges - species_edges,
            "below_minimum_threshold": species_edges - minimum_threshold_edges,
            "at_minimum_threshold": minimum_threshold_edges,
            "without_unique_gene_mapping": minimum_threshold_edges - mapped_edges,
            "mapped": mapped_edges,
            "removed_gene_self_loops": self_loops,
            "collapsed_duplicate_gene_edges": mapped_edges - self_loops - len(networks[thresholds[0]]),
            "output_at_minimum_threshold": len(networks[thresholds[0]]),
        },
        "unique_mapped_proteins": len(mapping),
        "ambiguous_proteins": int(ambiguous["string_protein_id"].nunique()),
        "unresolved_proteins": len(unresolved),
        "graph_summaries": graph_summaries,
        "primary_threshold": primary_threshold,
        "benchmark_gene_coverage": {
            "reference_genes": len(reference_gene_ids),
            "covered_genes": len(reference_gene_ids & primary_genes),
            "fraction": len(reference_gene_ids & primary_genes) / len(reference_gene_ids) if reference_gene_ids else None,
        },
        "core_case_gene_coverage": {
            disease_id: {
                "reference_genes": len(genes),
                "covered_genes": len(genes & primary_genes),
                "fraction": len(genes & primary_genes) / len(genes) if genes else None,
            }
            for disease_id, genes in sorted(core_gene_ids.items())
        },
        "threshold_edge_counts": {str(threshold): len(frame) for threshold, frame in networks.items()},
        "threshold_node_counts": {
            str(threshold): graph_summaries[str(threshold)]["nodes"] for threshold in thresholds
        },
    }
    return PpiResult(networks, mapping, unresolved.reset_index(drop=True), qc)


def write_ppi_outputs(result: PpiResult, paths: PpiOutputPaths, *, overwrite: bool = False) -> None:
    """Write all PPI artifacts after checking output conflicts as a group."""
    destinations = [*paths.networks.values(), paths.protein_mapping, paths.unresolved_proteins, paths.qc]
    existing = [path for path in destinations if Path(path).exists()]
    if existing and not overwrite:
        raise FileExistsError("Output files already exist; use overwrite=True: " + ", ".join(map(str, existing)))
    for path in destinations:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
    for threshold, frame in result.networks.items():
        frame.to_parquet(paths.networks[threshold], index=False)
    result.protein_mapping.to_parquet(paths.protein_mapping, index=False)
    result.unresolved_proteins.to_csv(paths.unresolved_proteins, sep="\t", index=False)
    Path(paths.qc).write_text(json.dumps(result.qc, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def default_ppi_output_paths(config: dict[str, Any]) -> PpiOutputPaths:
    """Build portable Milestone 3 output paths from configuration."""
    paths = config["paths"]
    processed, interim, outputs = Path(paths["processed"]), Path(paths["interim"]), Path(paths["outputs"])
    thresholds = tuple(int(value) for value in config["string"]["thresholds"])
    return PpiOutputPaths(
        networks={threshold: processed / f"string_gene_edges_{threshold}.parquet" for threshold in thresholds},
        protein_mapping=interim / "string_protein_gene_mapping.parquet",
        unresolved_proteins=interim / "unresolved_string_proteins.tsv",
        qc=outputs / "qc" / "ppi_qc.json",
    )


def _read_alias_mapping(path: Path, chunk_size: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    selected: list[pd.DataFrame] = []
    for chunk in pd.read_csv(path, sep="\t", dtype=str, chunksize=chunk_size):
        protein_column = "#string_protein_id" if "#string_protein_id" in chunk.columns else "string_protein_id"
        required = {protein_column, "alias", "source"}
        if not required.issubset(chunk.columns):
            raise ValueError(f"Missing STRING alias columns: {', '.join(sorted(required - set(chunk.columns)))}")
        mask = chunk["source"].eq(TRUSTED_NCBI_SOURCE) & chunk["alias"].str.fullmatch(r"\d+")
        selected.append(chunk.loc[mask, [protein_column, "alias", "source"]].rename(
            columns={protein_column: "string_protein_id", "alias": "gene_id", "source": "mapping_source"}
        ))
    aliases = pd.concat(selected, ignore_index=True).drop_duplicates()
    counts = aliases.groupby("string_protein_id")["gene_id"].nunique()
    ambiguous_ids = set(counts[counts > 1].index)
    ambiguous = aliases.loc[aliases["string_protein_id"].isin(ambiguous_ids)].copy()
    mapping = aliases.loc[~aliases["string_protein_id"].isin(ambiguous_ids)].sort_values(
        ["string_protein_id", "gene_id"], kind="stable"
    ).drop_duplicates("string_protein_id").reset_index(drop=True)
    return mapping, ambiguous.reset_index(drop=True)


def _graph_summary(frame: pd.DataFrame) -> dict[str, Any]:
    nodes = sorted(set(frame["gene_a"]) | set(frame["gene_b"]))
    if not nodes:
        return {"nodes": 0, "edges": 0, "largest_component_nodes": 0, "largest_component_fraction": 0.0,
                "top_degree_genes": []}
    node_index = {node: index for index, node in enumerate(nodes)}
    rows = frame["gene_a"].map(node_index).to_numpy()
    columns = frame["gene_b"].map(node_index).to_numpy()
    adjacency = coo_matrix(([1] * len(frame), (rows, columns)), shape=(len(nodes), len(nodes))).tocsr()
    component_count, labels = connected_components(adjacency, directed=False)
    sizes = pd.Series(labels).value_counts()
    degree = pd.concat([frame["gene_a"], frame["gene_b"]]).value_counts()
    top = [{"gene_id": str(gene_id), "degree": int(value)} for gene_id, value in degree.head(50).items()]
    largest = int(sizes.max()) if component_count else 0
    return {"nodes": len(nodes), "edges": len(frame), "components": int(component_count),
            "largest_component_nodes": largest, "largest_component_fraction": largest / len(nodes),
            "top_degree_genes": top}


def _source_info(path: Path, version: str) -> dict[str, Any]:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"file_name": path.name, "sha256": digest.hexdigest(), "bytes": path.stat().st_size,
            "version": version}