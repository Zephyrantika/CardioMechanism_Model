import pandas as pd
import pytest

from phenotype_network_v0.evaluation.bootstrap import (
    cluster_bootstrap_confidence_intervals,
    paired_cluster_bootstrap_difference,
)
from phenotype_network_v0.evaluation.ranking_metrics import (
    deterministic_score_order,
    evaluate_ordered_ranking,
    evaluate_positive_ranks,
)


def test_recall_mrr_multiple_positive_genes_and_candidate_coverage() -> None:
    result = evaluate_ordered_ranking(
        ("1", "2", "3", "4"),
        {"2", "4", "99"},
        recall_ks=(1, 2, 4),
    )
    assert result["reference_positive_count"] == 3
    assert result["covered_positive_count"] == 2
    assert result["candidate_coverage"] == pytest.approx(2 / 3)
    assert result["unevaluable_gene_ids"] == ["99"]
    assert result["recall_at_1"] == 0.0
    assert result["recall_at_2"] == pytest.approx(1 / 3)
    assert result["recall_at_4"] == pytest.approx(2 / 3)
    assert result["mrr"] == pytest.approx(0.5)
    assert result["average_precision"] == pytest.approx(1 / 3)
    assert result["auprc"] == pytest.approx(2 / 9)


def test_positive_rank_evaluation_matches_full_order() -> None:
    direct = evaluate_positive_ranks(
        (2, 4),
        candidate_gene_count=4,
        reference_positive_count=3,
        recall_ks=(1, 2, 4),
    )
    full = evaluate_ordered_ranking(
        ("1", "2", "3", "4"), {"2", "4", "99"}, recall_ks=(1, 2, 4)
    )
    for metric in (
        "recall_at_1", "recall_at_2", "recall_at_4",
        "mrr", "average_precision", "auprc",
    ):
        assert direct[metric] == pytest.approx(full[metric])


def test_empty_prediction_and_no_reference_handling() -> None:
    empty = evaluate_ordered_ranking((), {"10"}, recall_ks=(1, 5))
    assert not empty["evaluable"]
    assert empty["candidate_coverage"] == 0.0
    assert empty["recall_at_1"] == 0.0
    assert empty["mrr"] == 0.0
    assert empty["average_precision"] == 0.0
    assert empty["auprc"] == 0.0
    assert empty["unevaluable_gene_ids"] == ["10"]

    no_reference = evaluate_ordered_ranking(("10",), set(), recall_ks=(1,))
    assert no_reference["candidate_coverage"] is None
    assert no_reference["recall_at_1"] is None
    assert no_reference["mrr"] is None


def test_tied_scores_use_numeric_gene_id_order() -> None:
    order = deterministic_score_order(
        ("20", "3", "10", "2"),
        (0.5, 0.1, 0.5, 0.5),
    )
    assert order == ("2", "10", "20", "3")
    with pytest.raises(ValueError, match="unique"):
        deterministic_score_order(("1", "1"), (0.2, 0.1))


def _bootstrap_frame() -> pd.DataFrame:
    rows = []
    for model, offset in (("left", 0.2), ("right", 0.0)):
        rows.extend([
            {"disease_id": "D1", "disease_family_id": "F1", "model": model, "mrr": 0.1 + offset},
            {"disease_id": "D2", "disease_family_id": "F1", "model": model, "mrr": 0.2 + offset},
            {"disease_id": "D3", "disease_family_id": "F2", "model": model, "mrr": 0.3 + offset},
        ])
    return pd.DataFrame.from_records(rows)


def test_family_cluster_bootstrap_is_deterministic() -> None:
    frame = _bootstrap_frame()
    first = cluster_bootstrap_confidence_intervals(
        frame, metric_columns=("mrr",), iterations=200, seed=42
    )
    second = cluster_bootstrap_confidence_intervals(
        frame, metric_columns=("mrr",), iterations=200, seed=42
    )
    pd.testing.assert_frame_equal(first, second)
    assert set(first["model"]) == {"left", "right"}
    assert set(first["families"]) == {2}
    left = first.loc[first["model"].eq("left")].iloc[0]
    assert left["macro_mean"] == pytest.approx(0.4)
    assert left["ci_lower_95"] <= left["macro_mean"] <= left["ci_upper_95"]


def test_paired_cluster_bootstrap_difference() -> None:
    result = paired_cluster_bootstrap_difference(
        _bootstrap_frame(),
        "left",
        "right",
        metric="mrr",
        iterations=200,
        seed=42,
    )
    assert result["paired_diseases"] == 3
    assert result["families"] == 2
    assert result["mean_difference"] == pytest.approx(0.2)
    assert result["ci_lower_95"] == pytest.approx(0.2)
    assert result["ci_upper_95"] == pytest.approx(0.2)
