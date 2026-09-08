from pathlib import Path

import pandas as pd
import pronto
import pytest

from phenotype_network_v0.data.fold_builder import (
    assign_mondo_family_anchors,
    build_train_phenotype_gene_edges,
    make_disease_folds,
)

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"


def _inputs():
    diseases = [f"MONDO:T{i:02d}" for i in range(10)]
    samples = pd.DataFrame({"disease_id": diseases})
    phenotype_rows = []
    gene_rows = []
    for index, disease in enumerate(diseases):
        phenotype_rows.extend([
            {"disease_id": disease, "hpo_id": "HP:0000001", "frequency_weight": 1.0},
            {"disease_id": disease, "hpo_id": f"HP:{index + 2:07d}", "frequency_weight": 0.7},
        ])
        gene_rows.append({"disease_id": disease, "gene_id": str(index + 1)})
    phenotypes = pd.DataFrame(phenotype_rows)
    genes = pd.DataFrame(gene_rows)
    hpo_nodes = pd.DataFrame({"hpo_id": ["HP:0000001", *[f"HP:{i + 2:07d}" for i in range(10)]]})
    hpo_edges = pd.DataFrame({
        "child_hpo_id": [f"HP:{i + 2:07d}" for i in range(10)],
        "parent_hpo_id": ["HP:0000001"] * 10,
        "relation_type": ["is_a"] * 10,
    })
    string_edges = pd.DataFrame({"gene_a": [str(i) for i in range(1, 10)],
                                 "gene_b": [str(i + 1) for i in range(1, 10)]})
    reactome = pd.DataFrame({"gene_id": ["10"]})
    families = pd.DataFrame({
        "disease_id": diseases,
        "disease_family_id": [f"MONDO:F{i // 2}" for i in range(10)],
        "family_anchor_depth": [4] * 10,
        "family_anchor_candidates": [[f"MONDO:F{i // 2}"] for i in range(10)],
        "family_assignment_method": ["test"] * 10,
    })
    return samples, phenotypes, genes, hpo_nodes, hpo_edges, string_edges, reactome, families


def test_mondo_family_anchor_groups_related_diseases() -> None:
    ontology = pronto.Ontology(FIXTURES / "mondo_small.obo", encoding="utf-8")
    result = assign_mondo_family_anchors(
        {"MONDO:0021661", "MONDO:0005068"}, ontology, anchor_depth=1
    )
    assert result["disease_family_id"].nunique() == 1
    assert result.iloc[0]["disease_family_id"] == "MONDO:1000001"


def test_family_disjoint_folds_and_train_only_edges() -> None:
    result = make_disease_folds(*_inputs(), number_of_folds=5, seed=42,
                                parent_child_pairs={("MONDO:T00", "MONDO:T01")})
    assert result.qc["fold_disease_counts"] == {str(i): 2 for i in range(5)}
    assert result.qc["all_leakage_checks_passed"]
    assert len(result.candidate_universe) == 10
    for fold in result.folds:
        assert len(fold.train_diseases) == 6
        assert len(fold.validation_diseases) == 2
        assert len(fold.test_diseases) == 2
        contributors = {
            disease for values in fold.phenotype_gene_edges["contributing_disease_ids"] for disease in values
        }
        assert contributors <= set(fold.train_diseases)
        assert not contributors & set(fold.test_diseases)
        assert all(fold.leakage_report["checks"].values())
        assert fold.leakage_report["checks"]["no_cross_fold_parent_child_pairs"]


def test_train_ic_uses_only_training_diseases() -> None:
    result = make_disease_folds(*_inputs(), number_of_folds=5, seed=42)
    fold = result.folds[0]
    root = fold.hpo_ic.set_index("hpo_id").loc["HP:0000001"]
    assert root["annotation_disease_count"] == len(fold.train_diseases)
    assert root["ic"] == pytest.approx(0.0)
    test_specific = f"HP:{int(fold.test_diseases[0].split('T')[1]) + 2:07d}"
    assert fold.hpo_ic.set_index("hpo_id").loc[test_specific, "annotation_disease_count"] == 0


def test_phenotype_gene_weight_ablations_change_only_requested_factors() -> None:
    phenotypes = pd.DataFrame({
        "disease_id": ["D", "D"],
        "hpo_id": ["HP:1", "HP:2"],
        "frequency_weight": [0.25, 1.0],
    })
    genes = pd.DataFrame({"disease_id": ["D"], "gene_id": ["10"]})
    ic = pd.DataFrame({"hpo_id": ["HP:1", "HP:2"], "ic": [4.0, 2.0]})
    default = build_train_phenotype_gene_edges(
        {"D"}, phenotypes, genes, ic, 1.0
    ).set_index("hpo_id")
    no_frequency = build_train_phenotype_gene_edges(
        {"D"}, phenotypes, genes, ic, 1.0, use_frequency=False
    ).set_index("hpo_id")
    no_ic = build_train_phenotype_gene_edges(
        {"D"}, phenotypes, genes, ic, 1.0, use_ic=False
    ).set_index("hpo_id")
    assert default.loc["HP:1", "raw_weight"] < default.loc["HP:2", "raw_weight"]
    assert no_frequency.loc["HP:1", "raw_weight"] > no_frequency.loc["HP:2", "raw_weight"]
    assert no_ic.loc["HP:1", "raw_weight"] < no_ic.loc["HP:2", "raw_weight"]


def test_fold_assignment_is_deterministic() -> None:
    first = make_disease_folds(*_inputs(), number_of_folds=5, seed=7)
    second = make_disease_folds(*_inputs(), number_of_folds=5, seed=7)
    pd.testing.assert_frame_equal(first.assignments, second.assignments)
    for a, b in zip(first.folds, second.folds, strict=True):
        pd.testing.assert_frame_equal(a.phenotype_gene_edges, b.phenotype_gene_edges)


def test_fewer_than_five_families_fails() -> None:
    inputs = list(_inputs())
    inputs[-1] = inputs[-1].assign(disease_family_id="MONDO:ONE")
    with pytest.raises(ValueError, match="at least 5 disease families"):
        make_disease_folds(*inputs, number_of_folds=5)