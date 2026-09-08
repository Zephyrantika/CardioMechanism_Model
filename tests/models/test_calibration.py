import numpy as np
import pandas as pd
import pytest

from phenotype_network_v0.models.calibration import (
    CalibrationResult,
    build_hpo_context,
    calibrate_against_null_pool,
    calibration_config_hash,
    degree_correlations,
    match_null_profiles,
    rank_calibrated_query,
    summarize_phenotype_profiles,
)


def _context():
    edges = pd.DataFrame({
        "child_hpo_id": ["HP:A", "HP:B", "HP:X", "HP:Y", "HP:Y"],
        "parent_hpo_id": ["HP:ROOT", "HP:ROOT", "HP:A", "HP:A", "HP:B"],
    })
    return build_hpo_context(edges, ontology_root="HP:ROOT", abnormality_root="HP:ROOT")


def test_hpo_context_uses_global_categories_outside_abnormalities() -> None:
    edges = pd.DataFrame({
        "child_hpo_id": ["HP:ABN", "HP:MOD", "HP:CARDIO", "HP:TERM", "HP:ONSET"],
        "parent_hpo_id": ["HP:ALL", "HP:ALL", "HP:ABN", "HP:CARDIO", "HP:MOD"],
    })
    context = build_hpo_context(
        edges, ontology_root="HP:ALL", abnormality_root="HP:ABN"
    )
    assert context.depths["HP:TERM"] == 3
    assert context.systems["HP:TERM"] == ("HP:CARDIO",)
    assert context.systems["HP:ONSET"] == ("HP:MOD",)


def test_hpo_context_tracks_shortest_depth_and_multiple_systems() -> None:
    context = _context()
    assert context.top_level_systems == ("HP:A", "HP:B")
    assert context.depths["HP:X"] == 2
    assert context.depths["HP:Y"] == 2
    assert context.systems["HP:Y"] == ("HP:A", "HP:B")


def test_profile_summary_records_unresolved_terms() -> None:
    phenotypes = pd.DataFrame({
        "disease_id": ["D1", "D1"],
        "hpo_id": ["HP:X", "HP:OTHER"],
    })
    ic = pd.DataFrame({
        "hpo_id": ["HP:X", "HP:OTHER"],
        "ic": [2.0, 4.0],
    })
    profile = summarize_phenotype_profiles(("D1",), phenotypes, ic, _context()).iloc[0]
    assert profile["phenotype_count"] == 2
    assert profile["mean_depth"] == 2.0
    assert profile["ic_median"] == 3.0
    assert profile["top_level_system_ids"] == ("HP:A", "UNRESOLVED")
    assert profile["unresolved_depth_term_ids"] == ("HP:OTHER",)
    assert profile["unresolved_system_term_ids"] == ("HP:OTHER",)


def _profiles(ids):
    records = {
        "Q1": (5, 3.0, 1.0, 2.0, 3.0, 4.0, ("HP:A",)),
        "Q2": (9, 5.0, 1.5, 5.0, 6.0, 7.0, ("HP:B",)),
        "P1": (5, 3.0, 1.0, 2.0, 3.0, 4.0, ("HP:A",)),
        "P2": (6, 3.2, 1.1, 2.2, 3.2, 4.2, ("HP:A",)),
        "P3": (10, 5.0, 1.5, 5.0, 6.0, 7.0, ("HP:B",)),
    }
    return pd.DataFrame.from_records([
        {
            "disease_id": disease_id,
            "phenotype_count": records[disease_id][0],
            "mean_depth": records[disease_id][1],
            "std_depth": records[disease_id][2],
            "ic_q25": records[disease_id][3],
            "ic_median": records[disease_id][4],
            "ic_q75": records[disease_id][5],
            "top_level_system_ids": records[disease_id][6],
        }
        for disease_id in ids
    ])


def test_null_matching_is_deterministic_order_invariant_and_train_only() -> None:
    pool = _profiles(["P1", "P2", "P3"])
    first = match_null_profiles(
        _profiles(["Q1", "Q2"]), pool, null_queries=20, matched_pool_size=2,
        temperature=0.25, seed=42,
    )
    second = match_null_profiles(
        _profiles(["Q2", "Q1"]), pool, null_queries=20, matched_pool_size=2,
        temperature=0.25, seed=42,
    )
    columns = ["disease_id", "draw_index", "null_disease_id"]
    pd.testing.assert_frame_equal(
        first[columns].sort_values(columns[:2]).reset_index(drop=True),
        second[columns].sort_values(columns[:2]).reset_index(drop=True),
    )
    assert first.groupby("disease_id").size().to_dict() == {"Q1": 20, "Q2": 20}
    assert set(first["null_disease_id"]) <= {"P1", "P2", "P3"}
    assert set(first["source_split"]) == {"train"}
    assert set(first.loc[first["disease_id"].eq("Q1"), "null_disease_id"]) <= {"P1", "P2"}
    assert set(first.loc[first["disease_id"].eq("Q2"), "null_disease_id"]) <= {"P2", "P3"}


def test_calibration_statistics_and_zero_variance_audit() -> None:
    raw = np.asarray([0.5, 0.2])
    pool = np.asarray([[0.1, 0.3], [0.2, 0.2]])
    result = calibrate_against_null_pool(raw, pool, np.asarray([0, 1, 1, 1]))
    assert result.null_mean[0] == pytest.approx(0.25)
    assert result.null_std[0] == pytest.approx(np.sqrt(0.0075))
    assert result.z_score[0] == pytest.approx((0.5 - 0.25) / np.sqrt(0.0075))
    assert result.empirical_percentile.tolist() == [1.0, 0.5]
    assert result.zero_variance.tolist() == [False, True]
    assert result.z_score[1] == 0.0

    with pytest.raises(ValueError, match="outside"):
        calibrate_against_null_pool(raw, pool, np.asarray([2]))


def test_corrected_ranking_ties_and_degree_correlations_are_deterministic() -> None:
    calibrated = CalibrationResult(
        null_mean=np.asarray([0.1, 0.1, 0.1]),
        null_std=np.asarray([0.1, 0.1, 0.1]),
        z_score=np.asarray([1.0, 1.0, -1.0]),
        empirical_percentile=np.asarray([0.8, 0.8, 0.2]),
        zero_variance=np.asarray([False, False, False]),
    )
    config_hash = calibration_config_hash({"seed": 42})
    first = rank_calibrated_query(
        "D", ("20", "10", "30"), np.asarray([0.2, 0.2, 0.0]), calibrated,
        np.asarray([3, 2, 1]), np.asarray([4, 3, 2]),
        null_query_count=1000, calibration_hash=config_hash,
    )
    second = rank_calibrated_query(
        "D", ("20", "10", "30"), np.asarray([0.2, 0.2, 0.0]), calibrated,
        np.asarray([3, 2, 1]), np.asarray([4, 3, 2]),
        null_query_count=1000, calibration_hash=config_hash,
    )
    pd.testing.assert_frame_equal(first, second)
    assert first["gene_id"].tolist() == ["10", "20", "30"]
    assert first["corrected_rank"].tolist() == [1, 2, 3]
    assert first["calibration_hash"].nunique() == 1

    ranking = pd.DataFrame({
        "raw_rwr_score": [1.0, 2.0, 3.0, 4.0],
        "empirical_percentile": [4.0, 3.0, 2.0, 1.0],
        "z_score": [1.0, 2.0, 3.0, 4.0],
        "gene_degree": [1, 2, 3, 4],
    })
    correlations = degree_correlations(ranking)
    assert correlations["raw_score_gene_degree_spearman"] == pytest.approx(1.0)
    assert correlations["corrected_score_gene_degree_spearman"] == pytest.approx(-1.0)
    assert correlations["absolute_correlation_reduction"] == pytest.approx(0.0)
