import pandas as pd
import pytest

from phenotype_network_v0.models.semantic_baseline import (
    SemanticSimilarity,
    disease_similarities,
    phenotype_sets,
    rank_semantic_query,
    train_gene_disease_sets,
)


def _model():
    edges = pd.DataFrame({
        "child_hpo_id": ["HP:A", "HP:B", "HP:C"],
        "parent_hpo_id": ["HP:ROOT", "HP:A", "HP:ROOT"],
    })
    ic = pd.DataFrame({
        "hpo_id": ["HP:ROOT", "HP:A", "HP:B", "HP:C"],
        "ic": [0.0, 2.0, 3.0, 2.5],
    })
    return SemanticSimilarity(edges, ic)


def test_resnik_is_symmetric_and_uses_mica() -> None:
    model = _model()
    assert model.resnik("HP:B", "HP:A") == pytest.approx(2.0)
    assert model.resnik("HP:A", "HP:B") == pytest.approx(2.0)
    assert model.resnik("HP:B", "HP:B") == pytest.approx(3.0)
    assert model.resnik("HP:B", "HP:C") == pytest.approx(0.0)


def test_bma_identical_and_related_sets() -> None:
    model = _model()
    identical = model.bma({"HP:A", "HP:B"}, {"HP:A", "HP:B"})
    related = model.bma({"HP:A"}, {"HP:B"})
    unrelated = model.bma({"HP:B"}, {"HP:C"})
    assert identical > related > unrelated
    with pytest.raises(ValueError, match="non-empty"):
        model.bma(set(), {"HP:A"})


def test_semantic_ranking_is_complete_and_deterministic() -> None:
    model = _model()
    similarities = disease_similarities(
        {"HP:A"}, {"D1": {"HP:B"}, "D2": {"HP:C"}}, model
    )
    first = rank_semantic_query("Q", similarities, {"10": {"D1"}, "20": {"D2"}}, ("10", "20", "30"))
    second = rank_semantic_query("Q", similarities, {"10": {"D1"}, "20": {"D2"}}, ("10", "20", "30"))
    pd.testing.assert_frame_equal(first, second)
    assert len(first) == 9
    assert set(first["aggregation_method"]) == {"max", "top5_mean", "weighted_sum"}
    for _, group in first.groupby("aggregation_method"):
        assert group.iloc[0]["gene_id"] == "10"
        assert sorted(group["rank"]) == [1, 2, 3]
        assert set(group["gene_id"]) == {"10", "20", "30"}


def test_training_helpers_exclude_nontraining_labels() -> None:
    phenotypes = pd.DataFrame({"disease_id": ["train", "test"], "hpo_id": ["HP:A", "HP:C"]})
    genes = pd.DataFrame({"disease_id": ["train", "test"], "gene_id": ["10", "99"]})
    assert phenotype_sets(phenotypes, {"train"}) == {"train": {"HP:A"}}
    assert train_gene_disease_sets(genes, {"train"}) == {"10": {"train"}}