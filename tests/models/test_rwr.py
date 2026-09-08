import numpy as np
import pandas as pd
import pytest
from scipy.sparse import csr_matrix

from phenotype_network_v0.models.rwr import (
    build_seed_matrix,
    evaluate_gene_scores,
    rank_rwr_query,
    run_rwr,
    rwr_config_hash,
)


def _transition() -> csr_matrix:
    return csr_matrix([
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 0.0],
    ])


def test_alpha_one_returns_seed_exactly() -> None:
    seeds = np.asarray([[1.0, 0.25], [0.0, 0.75], [0.0, 0.0]])
    result = run_rwr(_transition(), seeds, alpha=1.0)
    np.testing.assert_array_equal(result.scores, seeds)
    np.testing.assert_array_equal(result.final_differences, [0.0, 0.0])
    assert result.converged


def test_rwr_converges_preserves_mass_and_is_deterministic() -> None:
    seeds = np.asarray([[1.0], [0.0], [0.0]])
    first = run_rwr(_transition(), seeds, alpha=0.3, tolerance=1e-12, max_iterations=500)
    second = run_rwr(_transition(), seeds, alpha=0.3, tolerance=1e-12, max_iterations=500)
    assert first.converged
    assert first.final_differences[0] < 1e-12
    assert first.scores.sum() == pytest.approx(1.0, abs=1e-12)
    assert np.all(first.scores >= 0)
    np.testing.assert_array_equal(first.scores, second.scores)
    assert first.iterations == second.iterations


def test_rwr_uses_row_transition_direction() -> None:
    transition = csr_matrix([[0.0, 1.0], [0.0, 1.0]])
    seed = np.asarray([1.0, 0.0])
    result = run_rwr(transition, seed, alpha=0.5, tolerance=1e-14)
    np.testing.assert_allclose(result.scores[:, 0], [0.5, 0.5], atol=1e-12)


def test_rwr_rejects_nonstochastic_or_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="row-stochastic"):
        run_rwr(csr_matrix([[0.0, 0.5], [0.0, 1.0]]), np.asarray([1.0, 0.0]))
    with pytest.raises(ValueError, match="nonnegative"):
        run_rwr(csr_matrix([[1.0, 0.0], [0.0, 1.0]]), np.asarray([1.1, -0.1]))


def test_seed_matrix_uses_frequency_times_ic_and_uniform_fallback() -> None:
    node_map = pd.DataFrame({
        "node_index": [0, 1, 2],
        "node_type": ["phenotype", "phenotype", "gene"],
        "node_id": ["HP:1", "HP:2", "10"],
    })
    phenotypes = pd.DataFrame({
        "disease_id": ["D1", "D1", "D2", "D2"],
        "hpo_id": ["HP:1", "HP:2", "HP:1", "HP:2"],
        "frequency_weight": [0.5, 1.0, 0.5, 1.0],
    })
    ic = pd.DataFrame({"hpo_id": ["HP:1", "HP:2"], "ic": [2.0, 4.0]})
    seeds, qc = build_seed_matrix(("D1",), phenotypes, ic, node_map)
    np.testing.assert_allclose(seeds[:, 0], [0.2, 0.8, 0.0])
    assert qc["uniform_seed_fallback_diseases"] == []

    zero_ic = ic.assign(ic=0.0)
    fallback, qc = build_seed_matrix(("D2",), phenotypes, zero_ic, node_map)
    np.testing.assert_allclose(fallback[:, 0], [0.5, 0.5, 0.0])
    assert qc["uniform_seed_fallback_diseases"] == ["D2"]


def test_seed_weight_ablations_remove_only_requested_factors() -> None:
    node_map = pd.DataFrame({
        "node_index": [0, 1],
        "node_type": ["phenotype", "phenotype"],
        "node_id": ["HP:1", "HP:2"],
    })
    phenotypes = pd.DataFrame({
        "disease_id": ["D", "D"],
        "hpo_id": ["HP:1", "HP:2"],
        "frequency_weight": [0.25, 1.0],
    })
    ic = pd.DataFrame({"hpo_id": ["HP:1", "HP:2"], "ic": [4.0, 2.0]})
    default, _ = build_seed_matrix(("D",), phenotypes, ic, node_map)
    no_frequency, qc = build_seed_matrix(
        ("D",), phenotypes, ic, node_map, use_frequency=False
    )
    no_ic, _ = build_seed_matrix(
        ("D",), phenotypes, ic, node_map, use_ic=False
    )
    np.testing.assert_allclose(default[:, 0], [1 / 3, 2 / 3])
    np.testing.assert_allclose(no_frequency[:, 0], [2 / 3, 1 / 3])
    np.testing.assert_allclose(no_ic[:, 0], [0.2, 0.8])
    assert not qc["use_frequency"] and qc["use_ic"]


def test_gene_evaluation_and_ranking_are_complete_and_deterministic() -> None:
    scores = np.asarray([[0.6], [0.3], [0.1]])
    metrics = evaluate_gene_scores(
        scores, ("D",), np.asarray([0, 1, 2]), ("20", "10", "30"), {"D": {"10"}}
    )
    assert metrics == {"evaluable_diseases": 1, "mrr": 0.5, "recall_at_10": 1.0}

    config_hash = rwr_config_hash({"alpha": 0.3})
    first = rank_rwr_query(
        "D", np.asarray([0.2, 0.2, 0.1]), ("20", "10", "30"), alpha=0.3,
        iterations=10, final_difference=1e-12, config_hash=config_hash,
    )
    second = rank_rwr_query(
        "D", np.asarray([0.2, 0.2, 0.1]), ("20", "10", "30"), alpha=0.3,
        iterations=10, final_difference=1e-12, config_hash=config_hash,
    )
    pd.testing.assert_frame_equal(first, second)
    assert first["gene_id"].tolist() == ["10", "20", "30"]
    assert first["rank"].tolist() == [1, 2, 3]
    assert first["config_hash"].nunique() == 1
