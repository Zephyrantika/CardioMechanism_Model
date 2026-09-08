"""Disease-family cluster bootstrap summaries."""

from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd


def cluster_bootstrap_confidence_intervals(
    per_disease: pd.DataFrame,
    *,
    model_column: str = "model",
    cluster_column: str = "disease_family_id",
    metric_columns: tuple[str, ...],
    iterations: int = 1000,
    seed: int = 42,
) -> pd.DataFrame:
    """Return model-wise macro means and family-cluster bootstrap 95% intervals."""
    if iterations < 1:
        raise ValueError("Bootstrap iterations must be positive")
    required = {model_column, cluster_column, *metric_columns}
    missing = required - set(per_disease.columns)
    if missing:
        raise ValueError(f"Bootstrap table lacks columns: {', '.join(sorted(missing))}")
    records = []
    for model, group in per_disease.groupby(model_column, sort=True):
        clusters = tuple(sorted(group[cluster_column].astype(str).unique()))
        if not clusters:
            continue
        cluster_frames = {
            cluster: group.loc[group[cluster_column].astype(str).eq(cluster)]
            for cluster in clusters
        }
        model_seed = int.from_bytes(
            hashlib.sha256(str(model).encode("utf-8")).digest()[:4], "little"
        )
        rng = np.random.default_rng(np.random.SeedSequence([seed, model_seed]))
        draws: dict[str, list[float]] = {metric: [] for metric in metric_columns}
        for _ in range(iterations):
            sampled = rng.choice(clusters, size=len(clusters), replace=True)
            sampled_frame = pd.concat(
                [cluster_frames[str(cluster)] for cluster in sampled],
                ignore_index=True,
            )
            for metric in metric_columns:
                values = pd.to_numeric(sampled_frame[metric], errors="coerce").dropna()
                draws[metric].append(float(values.mean()) if len(values) else np.nan)
        for metric in metric_columns:
            observed = pd.to_numeric(group[metric], errors="coerce").dropna()
            samples = np.asarray(draws[metric], dtype=float)
            samples = samples[np.isfinite(samples)]
            records.append({
                "model": str(model),
                "metric": metric,
                "diseases": len(group),
                "families": len(clusters),
                "macro_mean": float(observed.mean()) if len(observed) else None,
                "median": float(observed.median()) if len(observed) else None,
                "ci_lower_95": float(np.quantile(samples, 0.025)) if len(samples) else None,
                "ci_upper_95": float(np.quantile(samples, 0.975)) if len(samples) else None,
                "bootstrap_iterations": iterations,
                "seed": seed,
            })
    return pd.DataFrame.from_records(records)


def paired_cluster_bootstrap_difference(
    per_disease: pd.DataFrame,
    left_model: str,
    right_model: str,
    *,
    metric: str,
    disease_column: str = "disease_id",
    model_column: str = "model",
    cluster_column: str = "disease_family_id",
    iterations: int = 1000,
    seed: int = 42,
) -> dict[str, float | int | str]:
    """Estimate a paired left-minus-right difference using family clusters."""
    required = {disease_column, model_column, cluster_column, metric}
    missing = required - set(per_disease.columns)
    if missing:
        raise ValueError(f"Paired bootstrap table lacks columns: {', '.join(sorted(missing))}")
    selected = per_disease.loc[
        per_disease[model_column].isin([left_model, right_model]),
        [disease_column, model_column, cluster_column, metric],
    ]
    values = selected.pivot(index=disease_column, columns=model_column, values=metric)
    if left_model not in values or right_model not in values:
        raise ValueError("Both models must be present")
    families = (
        selected[[disease_column, cluster_column]]
        .drop_duplicates()
        .set_index(disease_column)[cluster_column]
        .astype(str)
    )
    paired = values[[left_model, right_model]].dropna()
    paired["difference"] = paired[left_model] - paired[right_model]
    paired["cluster"] = families.loc[paired.index]
    clusters = tuple(sorted(paired["cluster"].unique()))
    if not clusters:
        raise ValueError("No paired disease families are evaluable")
    groups = {
        cluster: paired.loc[paired["cluster"].eq(cluster), "difference"].to_numpy(dtype=float)
        for cluster in clusters
    }
    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(iterations):
        sampled = rng.choice(clusters, size=len(clusters), replace=True)
        values_sample = np.concatenate([groups[str(cluster)] for cluster in sampled])
        samples.append(float(values_sample.mean()))
    return {
        "left_model": left_model,
        "right_model": right_model,
        "metric": metric,
        "paired_diseases": len(paired),
        "families": len(clusters),
        "mean_difference": float(paired["difference"].mean()),
        "ci_lower_95": float(np.quantile(samples, 0.025)),
        "ci_upper_95": float(np.quantile(samples, 0.975)),
        "bootstrap_iterations": iterations,
        "seed": seed,
    }
