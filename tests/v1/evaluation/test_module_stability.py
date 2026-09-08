"""Module stability and audit tests (Milestone 5)."""

from __future__ import annotations

from phenotype_network_v1.evaluation.module_stability import (
    audit_module,
    extract_modules,
    jaccard,
    split_half_stability,
)

MEMBERSHIP = {
    "P1": {"100", "101", "102", "103", "104"},
    "P2": {"200"},
}


def test_jaccard() -> None:
    assert jaccard({"a", "b"}, {"b", "c"}) == 1 / 3
    assert jaccard(set(), set()) == 1.0


def test_audit_rejects_collapsed_and_hub() -> None:
    hub_scores = {"100": 9.0, "101": 0.1, "102": 0.1}
    reasons = audit_module(
        {"100", "101", "102"}, gene_scores=hub_scores, min_size=5
    )
    assert "collapsed" in reasons
    reasons_big = audit_module(
        {"100", "101", "102", "103", "104", "105", "106"},
        gene_scores={"100": 8.0, "101": 0.1, "102": 0.1, "103": 0.1, "104": 0.1, "105": 0.1, "106": 0.1},
        min_size=5, hub_fraction=0.8,
    )
    assert "single_hub_dominated" in reasons_big


def test_audit_retained_for_supported_spread_module() -> None:
    module = {"100", "101", "102", "103", "104"}
    scores = {gene: 0.2 for gene in module}
    reasons = audit_module(module, gene_scores=scores, min_size=5, max_size=100)
    assert reasons == []


def test_extract_modules_from_activation() -> None:
    modules = extract_modules(
        {"P1": 2.0},
        membership=MEMBERSHIP,
        gene_ids=["100", "101", "102", "103", "104", "200"],
        gene_scores=[1.0, 1.0, 0.0, 0.0, 0.0, 3.0],
        max_module_genes=100,
    )
    assert modules["P1"] == {"100", "101"}


def test_split_half_stability() -> None:
    module = {"100", "101", "102", "103", "104"}
    stability = split_half_stability(
        {"P1": module},
        ["100", "101", "102"], [1.0, 1.0, 1.0],
        ["103", "104", "105"], [1.0, 1.0, 1.0],
    )
    # Only genes present and positively scored count per half.
    assert 0.0 <= stability["P1"] <= 1.0
