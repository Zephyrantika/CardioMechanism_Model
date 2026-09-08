"""External-evidence isolation and support tests (Milestone 7)."""

from __future__ import annotations

from phenotype_network_v1.evaluation.external import (
    external_support_enrichment,
    record_mapping_rules,
)


def test_enrichment_fractions() -> None:
    ranked = ["g1", "g2", "g3", "g4", "g5"]
    external = {"gwas": {"g1", "g2", "g9"}}
    report = external_support_enrichment(ranked_gene_ids=ranked, top_k=5, external_gene_sets=external)
    assert report["gwas"]["in_top_k"] == 2
    assert report["gwas"]["fraction"] == 2 / 5


def test_top_k_limited() -> None:
    ranked = ["g1", "g2", "g3", "g4", "g5"]
    external = {"gtex": {"g1", "g2", "g3", "g4", "g5"}}
    report = external_support_enrichment(ranked_gene_ids=ranked, top_k=2, external_gene_sets=external)
    assert report["gtex"]["in_top_k"] == 2


def test_mapping_rules_recorded() -> None:
    rules = record_mapping_rules()
    assert "gwas_locus_to_gene" in rules
    assert "not causal proof" in rules["gtex_role"]
