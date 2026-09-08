import pandas as pd
import pytest

from phenotype_network_v0.models.modules import (
    detect_leiden_modules,
    match_module_stability,
    module_topology,
    overrepresentation_enrichment,
)


def _two_cliques() -> pd.DataFrame:
    records = []
    for genes in (("1", "2", "3", "4"), ("5", "6", "7", "8")):
        for left_index, left in enumerate(genes):
            for right in genes[left_index + 1:]:
                records.append({"gene_a": left, "gene_b": right, "weight": 1.0})
    records.append({"gene_a": "4", "gene_b": "5", "weight": 0.01})
    return pd.DataFrame.from_records(records)


def test_leiden_modules_are_deterministic_and_size_limited() -> None:
    edges = _two_cliques()
    first = detect_leiden_modules(
        tuple(map(str, range(1, 9))), edges, seed=42, minimum_size=3, maximum_size=5
    )
    second = detect_leiden_modules(
        tuple(map(str, range(1, 9))), edges, seed=42, minimum_size=3, maximum_size=5
    )
    assert first == second
    assert {frozenset(module) for module in first} == {
        frozenset({"1", "2", "3", "4"}),
        frozenset({"5", "6", "7", "8"}),
    }
    topology = module_topology(first[0], edges)
    assert topology["internal_edge_count"] == 6
    assert topology["internal_density"] == 1.0
    assert topology["single_hub_dominated"] is False


def test_module_stability_and_enrichment() -> None:
    stability, frequency = match_module_stability(
        ["1", "2", "3"],
        [
            [["1", "2", "3"], ["4", "5"]],
            [["1", "2"], ["3", "4"]],
        ],
    )
    assert stability == pytest.approx(5 / 6)
    assert frequency == {"1": 1.0, "2": 1.0, "3": 0.5}

    annotations = pd.DataFrame({
        "gene_id": ["1", "2", "3", "4"],
        "term_id": ["T1", "T1", "T2", "T2"],
        "term_name": ["one", "one", "two", "two"],
    })
    enrichment = overrepresentation_enrichment(
        ["1", "2"],
        annotations,
        {"1", "2", "3", "4"},
        term_column="term_id",
        name_column="term_name",
    )
    assert enrichment.iloc[0]["term_id"] == "T1"
    assert enrichment.iloc[0]["overlap_count"] == 2
    assert 0 <= enrichment.iloc[0]["fdr_bh"] <= 1