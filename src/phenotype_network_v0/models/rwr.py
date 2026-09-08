"""Sparse batched random walk with restart."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix


@dataclass(frozen=True)
class RwrResult:
    """Batched RWR scores and convergence diagnostics."""

    scores: np.ndarray
    iterations: int
    final_differences: np.ndarray
    converged: bool


def run_rwr(
    transition: csr_matrix,
    seeds: np.ndarray,
    *,
    alpha: float = 0.3,
    tolerance: float = 1e-10,
    max_iterations: int = 200,
) -> RwrResult:
    """Propagate column seed vectors over a row-stochastic transition matrix."""
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be between zero and one")
    if tolerance <= 0 or max_iterations < 1:
        raise ValueError("Invalid RWR convergence settings")
    matrix = transition.tocsr()
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Transition matrix must be square")
    if matrix.nnz and (not np.isfinite(matrix.data).all() or np.any(matrix.data < 0)):
        raise ValueError("Transition matrix must contain finite nonnegative weights")
    row_sums = np.asarray(matrix.sum(axis=1)).ravel()
    if not np.allclose(row_sums, 1.0, atol=1e-12, rtol=1e-12):
        raise ValueError("Transition matrix must be row-stochastic")
    seeds = np.asarray(seeds, dtype=float)
    if seeds.ndim == 1:
        seeds = seeds[:, None]
    if seeds.shape[0] != matrix.shape[0] or seeds.shape[1] == 0:
        raise ValueError("Transition and seed dimensions do not match")
    if not np.isfinite(seeds).all() or np.any(seeds < 0) or not np.allclose(
        seeds.sum(axis=0), 1.0, atol=1e-12, rtol=1e-12
    ):
        raise ValueError("Every seed vector must be nonnegative and sum to one")
    if alpha == 1.0:
        return RwrResult(seeds.copy(), 1, np.zeros(seeds.shape[1]), True)
    scores = seeds.copy()
    differences = np.full(seeds.shape[1], np.inf)
    propagation = matrix.transpose().tocsr()
    for iteration in range(1, max_iterations + 1):
        updated = alpha * seeds + (1.0 - alpha) * (propagation @ scores)
        differences = np.abs(updated - scores).sum(axis=0)
        scores = np.asarray(updated)
        if float(differences.max()) < tolerance:
            return RwrResult(scores, iteration, differences, True)
    return RwrResult(scores, max_iterations, differences, False)


def build_seed_matrix(
    disease_ids: tuple[str, ...],
    phenotypes: pd.DataFrame,
    hpo_ic: pd.DataFrame,
    node_map: pd.DataFrame,
    *,
    use_frequency: bool = True,
    use_ic: bool = True,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Build normalized frequency-times-train-IC HPO seed columns."""
    index = {
        str(row.node_id): int(row.node_index)
        for row in node_map.loc[node_map["node_type"].eq("phenotype")].itertuples(index=False)
    }
    ic = hpo_ic.set_index("hpo_id")["ic"].astype(float)
    selected = phenotypes.loc[phenotypes["disease_id"].isin(disease_ids)].groupby(
        ["disease_id", "hpo_id"], as_index=False
    )["frequency_weight"].max()
    groups = {str(disease): group for disease, group in selected.groupby("disease_id")}
    seeds = np.zeros((len(node_map), len(disease_ids)), dtype=float)
    uniform_fallback = []
    for column, disease_id in enumerate(disease_ids):
        group = groups.get(disease_id)
        if group is None or group.empty:
            raise ValueError(f"Query disease has no phenotypes: {disease_id}")
        term_ids = [str(value) for value in group["hpo_id"]]
        if any(term not in index or term not in ic.index for term in term_ids):
            raise ValueError(f"Query disease contains an HPO term absent from graph or IC: {disease_id}")
        frequency = (
            group["frequency_weight"].to_numpy(dtype=float)
            if use_frequency else np.ones(len(term_ids), dtype=float)
        )
        information = (
            ic.loc[term_ids].to_numpy(dtype=float)
            if use_ic else np.ones(len(term_ids), dtype=float)
        )
        weights = frequency * information
        if float(weights.sum()) <= 0:
            weights = np.ones(len(term_ids), dtype=float)
            uniform_fallback.append(disease_id)
        weights /= weights.sum()
        for term_id, weight in zip(term_ids, weights, strict=True):
            seeds[index[term_id], column] = weight
    return seeds, {
        "queries": len(disease_ids),
        "uniform_seed_fallback_diseases": uniform_fallback,
        "use_frequency": use_frequency,
        "use_ic": use_ic,
    }


def evaluate_gene_scores(
    scores: np.ndarray,
    disease_ids: tuple[str, ...],
    gene_node_indices: np.ndarray,
    candidate_gene_ids: tuple[str, ...],
    known_genes: dict[str, set[str]],
) -> dict[str, Any]:
    """Evaluate validation rankings without materializing full ranking tables."""
    gene_numbers = np.asarray([int(value) for value in candidate_gene_ids], dtype=np.int64)
    reciprocal_ranks = []
    recall10 = []
    evaluable = 0
    for column, disease_id in enumerate(disease_ids):
        positives = known_genes.get(disease_id, set()) & set(candidate_gene_ids)
        if not positives:
            continue
        values = scores[gene_node_indices, column]
        order = np.lexsort((gene_numbers, -values))
        ranks = {candidate_gene_ids[index]: rank for rank, index in enumerate(order, start=1)}
        positive_ranks = [ranks[gene] for gene in positives]
        reciprocal_ranks.append(1.0 / min(positive_ranks))
        recall10.append(sum(rank <= 10 for rank in positive_ranks) / len(positive_ranks))
        evaluable += 1
    return {"evaluable_diseases": evaluable,
            "mrr": float(np.mean(reciprocal_ranks)) if reciprocal_ranks else 0.0,
            "recall_at_10": float(np.mean(recall10)) if recall10 else 0.0}


def rank_rwr_query(
    disease_id: str,
    gene_scores: np.ndarray,
    candidate_gene_ids: tuple[str, ...],
    *,
    alpha: float,
    iterations: int,
    final_difference: float,
    config_hash: str,
) -> pd.DataFrame:
    """Create a deterministic complete candidate ranking for one query."""
    gene_numbers = np.asarray([int(value) for value in candidate_gene_ids], dtype=np.int64)
    order = np.lexsort((gene_numbers, -np.asarray(gene_scores, dtype=float)))
    return pd.DataFrame({
        "disease_id": disease_id,
        "gene_id": np.asarray(candidate_gene_ids, dtype=object)[order],
        "raw_rwr_score": np.asarray(gene_scores, dtype=float)[order],
        "rank": np.arange(1, len(candidate_gene_ids) + 1, dtype=np.int32),
        "alpha": alpha,
        "iterations": iterations,
        "final_diff": final_difference,
        "config_hash": config_hash,
    })


def rwr_config_hash(config: dict[str, Any]) -> str:
    """Hash the effective propagation configuration."""
    payload = json.dumps(config, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()