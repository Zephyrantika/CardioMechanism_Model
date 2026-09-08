"""Prepare frozen external validation tables without modifying the training graph."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd

from phenotype_network_v0.config import load_config
from phenotype_network_v0.data.external_validation import (
    parse_biogrid,
    parse_go_annotations,
    parse_go_terms,
    parse_gtex_expression,
    parse_gwas_catalog,
    unique_symbol_lookup,
)
from phenotype_network_v0.logging_utils import configure_logging


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    """Clean frozen validation sources into separately marked Parquet tables."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "configs/default.yaml",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logger = configure_logging(level=args.log_level, logger_name=__name__)
    config = load_config(args.config)
    processed = Path(config["paths"]["processed"])
    interim = Path(config["paths"]["interim"])
    outputs = Path(config["paths"]["outputs"])
    external = config["external_validation"]
    sources = {
        name: Path(external[f"{name}_file"])
        for name in ("biogrid", "gwas_catalog", "gtex", "go_ontology", "goa_human")
    }
    missing = [path for path in sources.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing frozen external sources: " + ", ".join(map(str, missing)))
    destinations = {
        "biogrid_edges": processed / "biogrid_gene_edges.parquet",
        "symbols": processed / "external_gene_symbol_mapping.parquet",
        "gwas": processed / "gwas_gene_support.parquet",
        "gtex": processed / "gtex_cardiovascular_expression.parquet",
        "go_annotations": processed / "go_gene_annotations.parquet",
        "go_terms": processed / "go_terms.parquet",
        "biogrid_unresolved": interim / "unresolved_biogrid_identifiers.tsv",
        "gwas_unresolved": interim / "unresolved_gwas_gene_mappings.tsv",
        "gtex_unresolved": interim / "unresolved_gtex_gene_mappings.tsv",
        "go_unresolved": interim / "unresolved_go_gene_mappings.tsv",
        "qc": outputs / "qc" / "external_validation_qc.json",
    }
    existing = [path for path in destinations.values() if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(
            "External validation outputs exist; use --overwrite: "
            + ", ".join(map(str, existing))
        )
    for path in destinations.values():
        path.parent.mkdir(parents=True, exist_ok=True)

    candidates = set(
        pd.read_parquet(Path(config["paths"]["folds"]) / "candidate_gene_universe.parquet")[
            "gene_id"
        ].astype(str)
    )
    edges, symbols, biogrid_unresolved, biogrid_qc = parse_biogrid(
        sources["biogrid"], candidates, version="5.0.260"
    )
    logger.info("Parsed BioGRID: edges=%d symbols=%d", len(edges), len(symbols))
    lookup = unique_symbol_lookup(symbols)
    gwas, gwas_unresolved, gwas_qc = parse_gwas_catalog(
        sources["gwas_catalog"], lookup, candidates
    )
    logger.info("Parsed GWAS Catalog: support_rows=%d", len(gwas))
    gtex, gtex_unresolved, gtex_qc = parse_gtex_expression(
        sources["gtex"], lookup, candidates
    )
    logger.info("Parsed GTEx: expression_rows=%d", len(gtex))
    go_annotations, go_unresolved, go_qc = parse_go_annotations(
        sources["goa_human"], lookup, candidates
    )
    go_terms = parse_go_terms(sources["go_ontology"], set(go_annotations["go_id"]))
    logger.info(
        "Parsed GOA: annotations=%d terms=%d", len(go_annotations), len(go_terms)
    )

    frames = {
        "biogrid_edges": edges,
        "symbols": symbols,
        "gwas": gwas,
        "gtex": gtex,
        "go_annotations": go_annotations,
        "go_terms": go_terms,
    }
    audits = {
        "biogrid_unresolved": biogrid_unresolved,
        "gwas_unresolved": gwas_unresolved,
        "gtex_unresolved": gtex_unresolved,
        "go_unresolved": go_unresolved,
    }
    temporary = {
        name: path.with_name(path.name + ".tmp")
        for name, path in destinations.items()
    }
    for path in temporary.values():
        if path.exists():
            path.unlink()
    for name, frame in frames.items():
        frame.to_parquet(temporary[name], index=False)
    for name, frame in audits.items():
        frame.to_csv(temporary[name], sep="\t", index=False)
    qc: dict[str, Any] = {
        "purpose": "external_validation_only",
        "entered_training_graph": False,
        "used_for_hyperparameter_selection": False,
        "expanded_candidate_universe": False,
        "candidate_genes": len(candidates),
        "sources": {
            name: {
                "path": str(path),
                "size_bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
            for name, path in sources.items()
        },
        "versions": {
            "biogrid": "5.0.260",
            "gwas_catalog": "2026-08-03 v1.0 full",
            "gtex": "v10 RNASeQC v2.4.2",
            "go": "releases/2026-06-15",
            "goa_human": "generated 2026-05-21",
        },
        "biogrid": biogrid_qc,
        "gwas_catalog": gwas_qc,
        "gtex": gtex_qc,
        "goa_human": go_qc,
        "go_terms": len(go_terms),
    }
    temporary["qc"].write_text(
        json.dumps(qc, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    for name, path in destinations.items():
        temporary[name].replace(path)
    logger.info("External validation preparation complete")


if __name__ == "__main__":
    main()