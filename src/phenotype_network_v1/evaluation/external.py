"""External-evidence evaluation for frozen V1 checkpoints (Milestone 7).

Frozen V1 checkpoints are evaluated against GWAS, GTEx, GO/GOA, BioGRID and
STRING-900 evidence without retraining. External sources never change
candidate membership or model weights. GTEx expression is tissue support, not
causal proof; GWAS support records its locus-to-gene mapping rule (the frozen
processed table already applied the rule; the rule name is recorded here).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


def load_external_gene_sets(processed_dir: str | Path) -> dict[str, set[str]]:
    """Load frozen external gene supports keyed by source name."""
    processed = Path(processed_dir)
    sets: dict[str, set[str]] = {}
    gwas = pd.read_parquet(processed / "gwas_gene_support.parquet")
    sets["gwas"] = {str(gene_id) for gene_id in gwas["gene_id"]}
    gtex = pd.read_parquet(processed / "gtex_cardiovascular_expression.parquet")
    sets["gtex_cardiovascular"] = {str(gene_id) for gene_id in gtex["gene_id"]}
    go = pd.read_parquet(processed / "go_gene_annotations.parquet")
    sets["go"] = {str(gene_id) for gene_id in go["gene_id"]}
    biogrid = pd.read_parquet(processed / "biogrid_gene_edges.parquet")
    sets["biogrid"] = {str(gene_id) for gene_id in set(biogrid["gene_a"]) | set(biogrid["gene_b"])}
    string900 = pd.read_parquet(processed / "string_gene_edges_900.parquet")
    sets["string900"] = {str(gene_id) for gene_id in set(string900["gene_a"]) | set(string900["gene_b"])}
    return sets


def external_support_enrichment(
    *,
    ranked_gene_ids: list[str],
    top_k: int,
    external_gene_sets: dict[str, set[str]],
) -> dict[str, Any]:
    """Fraction of external-supported genes inside the model top-k.

    Only candidates ranked by the model are considered; enrichment is the
    top-k support fraction per source. This measures external agreement and
    never modifies candidate membership or weights.
    """
    top = ranked_gene_ids[:top_k]
    return {
        source: {
            "in_top_k": int(sum(gene_id in genes for gene_id in top)),
            "fraction": float(
                sum(gene_id in genes for gene_id in top) / max(1, len(top))
            ),
        }
        for source, genes in external_gene_sets.items()
    }


def record_mapping_rules() -> dict[str, str]:
    """Record the frozen external mapping rules for audit."""
    return {
        "gwas_locus_to_gene": "frozen gwas_gene_support.parquet (rule applied at "
        "V0 external preparation; configs/data.yaml gwas_catalog_version 2026-08-03)",
        "gtex_role": "tissue expression support only - not causal proof",
        "go_goa": "frozen go_gene_annotations.parquet (v0 external preparation)",
        "biogrid": "physical interactions, frozen (v0 external preparation)",
        "string900": "frozen 900-threshold PPI sensitivity set",
    }


__all__ = [
    "external_support_enrichment",
    "load_external_gene_sets",
    "record_mapping_rules",
]
