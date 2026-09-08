"""Matched-null calibration for phenotype-seeded RWR scores."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class HpoContext:
    """Shortest HPO depths and top-level phenotypic systems."""

    depths: dict[str, int]
    systems: dict[str, tuple[str, ...]]
    top_level_systems: tuple[str, ...]


@dataclass(frozen=True)
class CalibrationResult:
    """Per-gene matched-null summary statistics."""

    null_mean: np.ndarray
    null_std: np.ndarray
    z_score: np.ndarray
    empirical_percentile: np.ndarray
    zero_variance: np.ndarray


def build_hpo_context(
    hpo_edges: pd.DataFrame,
    *,
    ontology_root: str = "HP:0000001",
    abnormality_root: str = "HP:0000118",
) -> HpoContext:
    """Compute global HPO depth and hybrid top-level clinical systems."""
    required = {"child_hpo_id", "parent_hpo_id"}
    missing = required - set(hpo_edges.columns)
    if missing:
        raise ValueError(f"HPO edges lack columns: {', '.join(sorted(missing))}")
    children: dict[str, set[str]] = defaultdict(set)
    for row in hpo_edges.itertuples(index=False):
        children[str(row.parent_hpo_id)].add(str(row.child_hpo_id))
    global_categories = set(children.get(ontology_root, set()))
    if not global_categories:
        raise ValueError(f"HPO ontology root has no children: {ontology_root}")
    abnormality_systems = set(children.get(abnormality_root, set()))
    if abnormality_root != ontology_root and not abnormality_systems:
        raise ValueError(f"HPO abnormality root has no children: {abnormality_root}")
    system_roots = (
        abnormality_systems | (global_categories - {abnormality_root})
        if abnormality_root != ontology_root
        else global_categories
    )
    top_level = tuple(sorted(system_roots))

    depths = {ontology_root: 0}
    queue = deque([ontology_root])
    while queue:
        parent = queue.popleft()
        candidate_depth = depths[parent] + 1
        for child in sorted(children.get(parent, set())):
            if child not in depths or candidate_depth < depths[child]:
                depths[child] = candidate_depth
                queue.append(child)

    system_sets: dict[str, set[str]] = defaultdict(set)
    system_queue = deque((system, system) for system in top_level)
    while system_queue:
        node, system = system_queue.popleft()
        if system in system_sets[node]:
            continue
        system_sets[node].add(system)
        for child in sorted(children.get(node, set())):
            system_queue.append((child, system))
    systems = {term: tuple(sorted(values)) for term, values in system_sets.items()}
    return HpoContext(depths=depths, systems=systems, top_level_systems=top_level)


def summarize_phenotype_profiles(
    disease_ids: tuple[str, ...],
    phenotypes: pd.DataFrame,
    hpo_ic: pd.DataFrame,
    context: HpoContext,
) -> pd.DataFrame:
    """Summarize phenotype count, depth, IC distribution, and HPO systems."""
    if len(set(disease_ids)) != len(disease_ids):
        raise ValueError("Disease identifiers must be unique")
    ic = hpo_ic.set_index("hpo_id")["ic"].astype(float)
    selected = phenotypes.loc[phenotypes["disease_id"].astype(str).isin(disease_ids)]
    groups = {
        str(disease): tuple(sorted(set(group["hpo_id"].astype(str))))
        for disease, group in selected.groupby("disease_id")
    }
    records = []
    for disease_id in disease_ids:
        term_ids = groups.get(disease_id, ())
        if not term_ids:
            raise ValueError(f"Disease has no phenotype profile: {disease_id}")
        missing_ic = sorted(term for term in term_ids if term not in ic.index)
        if missing_ic:
            raise ValueError(f"Disease has HPO terms absent from train IC: {disease_id}")
        unresolved_depth = tuple(term for term in term_ids if term not in context.depths)
        resolved_depths = np.asarray(
            [context.depths[term] for term in term_ids if term in context.depths],
            dtype=float,
        )
        if not len(resolved_depths):
            raise ValueError(f"Disease has no HPO terms reachable from abnormality root: {disease_id}")
        unresolved_system = tuple(term for term in term_ids if term not in context.systems)
        system_ids = {
            system
            for term in term_ids
            for system in context.systems.get(term, ())
        }
        if unresolved_system:
            system_ids.add("UNRESOLVED")
        ic_values = ic.loc[list(term_ids)].to_numpy(dtype=float)
        records.append({
            "disease_id": disease_id,
            "phenotype_count": len(term_ids),
            "mean_depth": float(resolved_depths.mean()),
            "std_depth": float(resolved_depths.std()),
            "ic_q25": float(np.quantile(ic_values, 0.25)),
            "ic_median": float(np.median(ic_values)),
            "ic_q75": float(np.quantile(ic_values, 0.75)),
            "mean_ic": float(ic_values.mean()),
            "std_ic": float(ic_values.std()),
            "top_level_system_ids": tuple(sorted(system_ids)),
            "top_level_system_count": len(system_ids),
            "unresolved_depth_term_ids": unresolved_depth,
            "unresolved_system_term_ids": unresolved_system,
        })
    return pd.DataFrame.from_records(records)


def match_null_profiles(
    query_profiles: pd.DataFrame,
    pool_profiles: pd.DataFrame,
    *,
    null_queries: int,
    matched_pool_size: int,
    temperature: float,
    seed: int,
) -> pd.DataFrame:
    """Bootstrap deterministic null queries from nearest training profiles."""
    if null_queries < 1 or matched_pool_size < 1 or temperature <= 0:
        raise ValueError("Invalid null matching settings")
    if query_profiles.empty or pool_profiles.empty:
        raise ValueError("Query and null profile tables must be non-empty")
    required = {
        "disease_id", "phenotype_count", "mean_depth", "std_depth",
        "ic_q25", "ic_median", "ic_q75", "top_level_system_ids",
    }
    for name, frame in (("query", query_profiles), ("pool", pool_profiles)):
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"{name} profiles lack columns: {', '.join(sorted(missing))}")
        if frame["disease_id"].astype(str).duplicated().any():
            raise ValueError(f"{name} profile disease identifiers are not unique")

    pool = pool_profiles.copy()
    pool["disease_id"] = pool["disease_id"].astype(str)
    pool = pool.sort_values("disease_id", kind="stable").reset_index(drop=True)
    count_values = np.log1p(pool["phenotype_count"].to_numpy(dtype=float))
    depth_values = pool[["mean_depth", "std_depth"]].to_numpy(dtype=float)
    ic_values = pool[["ic_q25", "ic_median", "ic_q75"]].to_numpy(dtype=float)
    count_scale = max(float(count_values.std()), 0.25)
    depth_scale = np.maximum(depth_values.std(axis=0), 0.25)
    ic_scale = np.maximum(ic_values.std(axis=0), 0.25)

    records = []
    for query in query_profiles.sort_values("disease_id", kind="stable").itertuples(index=False):
        query_id = str(query.disease_id)
        eligible = pool.loc[pool["disease_id"].ne(query_id)].reset_index(drop=True)
        if eligible.empty:
            raise ValueError(f"No eligible null profiles for query: {query_id}")
        eligible_count = np.log1p(eligible["phenotype_count"].to_numpy(dtype=float))
        count_distance = np.abs(eligible_count - np.log1p(float(query.phenotype_count))) / count_scale
        eligible_depth = eligible[["mean_depth", "std_depth"]].to_numpy(dtype=float)
        query_depth = np.asarray([query.mean_depth, query.std_depth], dtype=float)
        depth_distance = np.mean(np.abs(eligible_depth - query_depth) / depth_scale, axis=1)
        eligible_ic = eligible[["ic_q25", "ic_median", "ic_q75"]].to_numpy(dtype=float)
        query_ic = np.asarray([query.ic_q25, query.ic_median, query.ic_q75], dtype=float)
        ic_distance = np.mean(np.abs(eligible_ic - query_ic) / ic_scale, axis=1)
        query_systems = set(query.top_level_system_ids)
        system_distance = np.asarray([
            _jaccard_distance(query_systems, set(values))
            for values in eligible["top_level_system_ids"]
        ])
        total_distance = count_distance + depth_distance + ic_distance + system_distance
        order = np.lexsort((eligible["disease_id"].to_numpy(dtype=str), total_distance))
        size = min(matched_pool_size, len(order))
        nearest = order[:size]
        nearest_distance = total_distance[nearest]
        weights = np.exp(-(nearest_distance - nearest_distance.min()) / temperature)
        probabilities = weights / weights.sum()
        disease_seed = int.from_bytes(hashlib.sha256(query_id.encode("utf-8")).digest()[:4], "little")
        rng = np.random.default_rng(np.random.SeedSequence([seed, disease_seed]))
        draws = rng.choice(size, size=null_queries, replace=True, p=probabilities)
        for draw_index, local_index in enumerate(draws):
            pool_index = int(nearest[int(local_index)])
            records.append({
                "disease_id": query_id,
                "draw_index": draw_index,
                "null_disease_id": str(eligible.iloc[pool_index]["disease_id"]),
                "match_rank": int(local_index) + 1,
                "sampling_probability": float(probabilities[int(local_index)]),
                "total_match_distance": float(total_distance[pool_index]),
                "phenotype_count_distance": float(count_distance[pool_index]),
                "depth_distance": float(depth_distance[pool_index]),
                "ic_distance": float(ic_distance[pool_index]),
                "system_distance": float(system_distance[pool_index]),
                "source_split": "train",
            })
    return pd.DataFrame.from_records(records)


def calibrate_against_null_pool(
    raw_scores: np.ndarray,
    null_pool_scores: np.ndarray,
    sampled_pool_indices: np.ndarray,
    *,
    zero_variance_tolerance: float = 1e-12,
) -> CalibrationResult:
    """Calculate bootstrap null moments, Z-scores, and empirical percentiles."""
    raw = np.asarray(raw_scores, dtype=float)
    pool = np.asarray(null_pool_scores, dtype=float)
    sampled = np.asarray(sampled_pool_indices, dtype=int)
    if raw.ndim != 1 or pool.ndim != 2 or pool.shape[0] != len(raw):
        raise ValueError("Raw and null score dimensions do not match")
    if sampled.ndim != 1 or not len(sampled):
        raise ValueError("At least one sampled null index is required")
    if sampled.min() < 0 or sampled.max() >= pool.shape[1]:
        raise ValueError("Sampled null index is outside the pool")
    if zero_variance_tolerance < 0:
        raise ValueError("zero_variance_tolerance must be nonnegative")
    if not np.isfinite(raw).all() or not np.isfinite(pool).all():
        raise ValueError("Scores must be finite")
    counts = np.bincount(sampled, minlength=pool.shape[1]).astype(np.int64)
    probabilities = counts / counts.sum()
    null_mean = np.sum(pool * probabilities[None, :], axis=1)
    second_moment = np.sum(np.square(pool) * probabilities[None, :], axis=1)
    variance = np.maximum(second_moment - np.square(null_mean), 0.0)
    null_std = np.sqrt(variance)
    zero_variance = null_std <= zero_variance_tolerance
    z_score = np.zeros_like(raw)
    np.divide(raw - null_mean, null_std, out=z_score, where=~zero_variance)
    less_counts = np.sum(
        np.less(pool, raw[:, None]) * counts[None, :], axis=1
    )
    equal_counts = np.sum(
        np.equal(pool, raw[:, None]) * counts[None, :], axis=1
    )
    empirical_percentile = (less_counts + 0.5 * equal_counts) / counts.sum()
    return CalibrationResult(null_mean, null_std, z_score, empirical_percentile, zero_variance)


def rank_calibrated_query(
    disease_id: str,
    candidate_gene_ids: tuple[str, ...],
    raw_scores: np.ndarray,
    calibrated: CalibrationResult,
    gene_degree: np.ndarray,
    graph_degree: np.ndarray,
    *,
    null_query_count: int,
    calibration_hash: str,
) -> pd.DataFrame:
    """Create a complete deterministic ranking from calibrated scores."""
    arrays = [
        np.asarray(raw_scores, dtype=float),
        calibrated.null_mean,
        calibrated.null_std,
        calibrated.z_score,
        calibrated.empirical_percentile,
        calibrated.zero_variance,
        np.asarray(gene_degree, dtype=np.int64),
        np.asarray(graph_degree, dtype=np.int64),
    ]
    if any(len(values) != len(candidate_gene_ids) for values in arrays):
        raise ValueError("Candidate and calibration array lengths differ")
    gene_numbers = np.asarray([int(value) for value in candidate_gene_ids], dtype=np.int64)
    raw_order = np.lexsort((gene_numbers, -arrays[0]))
    raw_rank = np.empty(len(candidate_gene_ids), dtype=np.int32)
    raw_rank[raw_order] = np.arange(1, len(candidate_gene_ids) + 1, dtype=np.int32)
    order = np.lexsort((
        gene_numbers,
        -arrays[0],
        -calibrated.z_score,
        -calibrated.empirical_percentile,
    ))
    return pd.DataFrame({
        "disease_id": disease_id,
        "gene_id": np.asarray(candidate_gene_ids, dtype=object)[order],
        "raw_rwr_score": arrays[0][order],
        "raw_rank": raw_rank[order],
        "null_mean": calibrated.null_mean[order],
        "null_std": calibrated.null_std[order],
        "z_score": calibrated.z_score[order],
        "empirical_percentile": calibrated.empirical_percentile[order],
        "corrected_rank": np.arange(1, len(candidate_gene_ids) + 1, dtype=np.int32),
        "gene_degree": arrays[6][order],
        "graph_degree": arrays[7][order],
        "zero_null_variance": calibrated.zero_variance[order],
        "null_query_count": null_query_count,
        "calibration_hash": calibration_hash,
    })


def degree_correlations(ranking: pd.DataFrame) -> dict[str, float | None]:
    """Measure raw and calibrated score association with PPI degree."""
    raw = _spearman(ranking["raw_rwr_score"], ranking["gene_degree"])
    corrected = _spearman(ranking["empirical_percentile"], ranking["gene_degree"])
    z_score = _spearman(ranking["z_score"], ranking["gene_degree"])
    return {
        "raw_score_gene_degree_spearman": raw,
        "corrected_score_gene_degree_spearman": corrected,
        "z_score_gene_degree_spearman": z_score,
        "absolute_correlation_reduction": (
            abs(raw) - abs(corrected) if raw is not None and corrected is not None else None
        ),
    }


def calibration_config_hash(config: dict[str, Any]) -> str:
    """Hash the effective calibration settings and upstream identities."""
    payload = json.dumps(config, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _jaccard_distance(left: set[str], right: set[str]) -> float:
    union = left | right
    return 1.0 - len(left & right) / len(union) if union else 0.0


def _spearman(left: pd.Series, right: pd.Series) -> float | None:
    if left.nunique(dropna=True) < 2 or right.nunique(dropna=True) < 2:
        return None
    left_rank = left.rank(method="average").to_numpy(dtype=float)
    right_rank = right.rank(method="average").to_numpy(dtype=float)
    left_centered = left_rank - left_rank.mean()
    right_centered = right_rank - right_rank.mean()
    denominator = float(np.sqrt(
        np.sum(np.square(left_centered)) * np.sum(np.square(right_centered))
    ))
    if denominator == 0:
        return None
    value = float(np.sum(left_centered * right_centered) / denominator)
    return value if np.isfinite(value) else None
