"""Parse the Human Phenotype Ontology into auditable tabular outputs."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path
from typing import Any

import pandas as pd
import pronto

NODE_COLUMNS = [
    "hpo_id",
    "hpo_name",
    "definition",
    "synonyms",
    "is_obsolete",
    "replaced_by",
]
EDGE_COLUMNS = ["child_hpo_id", "parent_hpo_id", "relation_type"]
MAPPING_COLUMNS = [
    "original_hpo_id",
    "original_hpo_name",
    "replacement_hpo_id",
    "replacement_hpo_name",
    "mapping_method",
]
UNRESOLVED_COLUMNS = ["hpo_id", "hpo_name", "reason", "replacement_candidates"]


@dataclass(frozen=True)
class HpoParseResult:
    """In-memory HPO tables and quality-control summary."""

    nodes: pd.DataFrame
    edges: pd.DataFrame
    obsolete_mapping: pd.DataFrame
    unresolved_terms: pd.DataFrame
    qc: dict[str, Any]


@dataclass(frozen=True)
class HpoOutputPaths:
    """Locations of all Milestone 1 output artifacts."""

    nodes: Path
    edges: Path
    obsolete_mapping: Path
    unresolved_terms: Path
    qc: Path


def parse_hpo_ontology(input_path: Path) -> HpoParseResult:
    """Parse an OBO file, retaining active HPO terms and auditing obsolete IDs."""
    source = Path(input_path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"HPO ontology file does not exist: {source}")

    ontology = pronto.Ontology(source, encoding="utf-8")
    hpo_terms = {term.id: term for term in ontology.terms() if term.id.startswith("HP:")}
    active_terms = {term_id: term for term_id, term in hpo_terms.items() if not term.obsolete}

    node_records = [
        {
            "hpo_id": term.id,
            "hpo_name": term.name or "",
            "definition": term.definition or "",
            "synonyms": sorted({synonym.description for synonym in term.synonyms}),
            "is_obsolete": False,
            "replaced_by": None,
        }
        for term in active_terms.values()
    ]

    mapping_records: list[dict[str, Any]] = []
    unresolved_records: list[dict[str, Any]] = []
    replacement_lookup: dict[str, str] = {}
    for term in hpo_terms.values():
        if not term.obsolete:
            continue
        replacement_id, reason, candidates = _resolve_replacement(term, hpo_terms)
        if replacement_id is None:
            unresolved_records.append(
                {
                    "hpo_id": term.id,
                    "hpo_name": term.name or "",
                    "reason": reason,
                    "replacement_candidates": candidates,
                }
            )
            continue
        replacement = active_terms[replacement_id]
        replacement_lookup[term.id] = replacement_id
        mapping_records.append(
            {
                "original_hpo_id": term.id,
                "original_hpo_name": term.name or "",
                "replacement_hpo_id": replacement_id,
                "replacement_hpo_name": replacement.name or "",
                "mapping_method": "unique_replaced_by",
            }
        )

    edge_records: list[dict[str, str]] = []
    input_edge_count = 0
    unresolved_edge_count = 0
    for term in active_terms.values():
        for parent in term.superclasses(distance=1, with_self=False):
            if not parent.id.startswith("HP:"):
                continue
            input_edge_count += 1
            parent_id = replacement_lookup.get(parent.id, parent.id)
            if parent_id not in active_terms:
                unresolved_edge_count += 1
                unresolved_records.append(
                    {
                        "hpo_id": term.id,
                        "hpo_name": term.name or "",
                        "reason": "is_a_parent_not_active",
                        "replacement_candidates": [parent.id],
                    }
                )
                continue
            edge_records.append(
                {
                    "child_hpo_id": term.id,
                    "parent_hpo_id": parent_id,
                    "relation_type": "is_a",
                }
            )

    nodes = _frame(node_records, NODE_COLUMNS, ["hpo_id"])
    raw_edges = pd.DataFrame.from_records(edge_records, columns=EDGE_COLUMNS)
    duplicate_edge_count = int(raw_edges.duplicated(subset=EDGE_COLUMNS).sum())
    edges = _frame(edge_records, EDGE_COLUMNS, EDGE_COLUMNS)
    obsolete_mapping = _frame(mapping_records, MAPPING_COLUMNS, ["original_hpo_id"])
    unresolved_terms = _frame(
        unresolved_records,
        UNRESOLVED_COLUMNS,
        ["hpo_id", "reason"],
    )

    node_ids = set(nodes["hpo_id"])
    invalid_edges = edges.loc[
        ~edges["child_hpo_id"].isin(node_ids) | ~edges["parent_hpo_id"].isin(node_ids)
    ]
    if not invalid_edges.empty:
        raise ValueError("Parsed HPO edges reference nodes absent from the final node table")

    obsolete_count = sum(term.obsolete for term in hpo_terms.values())
    qc: dict[str, Any] = {
        "source_file": str(source),
        "source_sha256": _sha256(source),
        "ontology_data_version": ontology.metadata.data_version,
        "ontology_format_version": ontology.metadata.format_version,
        "source_encoding": "utf-8",
        "parser": f"pronto {version('pronto')}",
        "input_records": len(hpo_terms),
        "output_records": len(nodes),
        "removed_records": obsolete_count,
        "duplicate_records": duplicate_edge_count,
        "unmapped_records": len(unresolved_terms),
        "obsolete_records": obsolete_count,
        "replaced_records": len(obsolete_mapping),
        "unresolved_records": len(unresolved_terms),
        "input_is_a_edges": input_edge_count,
        "output_is_a_edges": len(edges),
        "unresolved_is_a_edges": unresolved_edge_count,
        "non_hpo_terms_ignored": len(ontology) - len(hpo_terms),
    }
    return HpoParseResult(nodes, edges, obsolete_mapping, unresolved_terms, qc)


def write_hpo_outputs(
    result: HpoParseResult,
    paths: HpoOutputPaths,
    *,
    overwrite: bool = False,
) -> None:
    """Write all HPO artifacts after checking output conflicts as a group."""
    destinations = [
        paths.nodes,
        paths.edges,
        paths.obsolete_mapping,
        paths.unresolved_terms,
        paths.qc,
    ]
    existing = [Path(path) for path in destinations if Path(path).exists()]
    if existing and not overwrite:
        joined = ", ".join(str(path) for path in existing)
        raise FileExistsError(f"Output files already exist; use overwrite=True: {joined}")
    for path in destinations:
        Path(path).parent.mkdir(parents=True, exist_ok=True)

    result.nodes.to_parquet(paths.nodes, index=False)
    result.edges.to_parquet(paths.edges, index=False)
    result.obsolete_mapping.to_parquet(paths.obsolete_mapping, index=False)
    unresolved = result.unresolved_terms.copy()
    unresolved["replacement_candidates"] = unresolved["replacement_candidates"].map(json.dumps)
    unresolved.to_csv(paths.unresolved_terms, sep="\t", index=False)
    Path(paths.qc).write_text(
        json.dumps(result.qc, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )


def default_hpo_output_paths(config: dict[str, Any]) -> HpoOutputPaths:
    """Build the standard Milestone 1 output paths from a loaded config."""
    paths = config.get("paths")
    if not isinstance(paths, dict):
        raise ValueError("Configuration must contain a 'paths' mapping")
    required = ("processed", "interim", "outputs")
    missing = [name for name in required if name not in paths]
    if missing:
        raise ValueError(f"Configuration is missing path keys: {', '.join(missing)}")
    processed = Path(paths["processed"])
    interim = Path(paths["interim"])
    outputs = Path(paths["outputs"])
    return HpoOutputPaths(
        nodes=processed / "hpo_nodes.parquet",
        edges=processed / "hpo_edges.parquet",
        obsolete_mapping=interim / "obsolete_hpo_mapping.parquet",
        unresolved_terms=interim / "unresolved_hpo_terms.tsv",
        qc=outputs / "qc" / "hpo_qc.json",
    )


def _resolve_replacement(
    term: pronto.Term,
    hpo_terms: dict[str, pronto.Term],
) -> tuple[str | None, str, list[str]]:
    current = term
    visited = {term.id}
    while current.obsolete:
        candidates = sorted(replacement.id for replacement in current.replaced_by)
        if not candidates:
            return None, "obsolete_without_replacement", candidates
        if len(candidates) != 1:
            return None, "obsolete_with_ambiguous_replacement", candidates
        replacement_id = candidates[0]
        if replacement_id in visited:
            return None, "obsolete_replacement_cycle", candidates
        replacement = hpo_terms.get(replacement_id)
        if replacement is None:
            return None, "replacement_not_in_hpo", candidates
        visited.add(replacement_id)
        current = replacement
    return current.id, "", [current.id]


def _frame(
    records: list[dict[str, Any]],
    columns: list[str],
    sort_columns: list[str],
) -> pd.DataFrame:
    frame = pd.DataFrame.from_records(records, columns=columns)
    if frame.empty:
        return frame
    return (
        frame.drop_duplicates(subset=sort_columns)
        .sort_values(sort_columns, kind="stable")
        .reset_index(drop=True)
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

