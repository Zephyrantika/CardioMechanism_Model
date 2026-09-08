"""Calibrate fold-specific RWR scores with matched training-profile null queries."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.sparse import load_npz

from phenotype_network_v0.config import load_config
from phenotype_network_v0.logging_utils import configure_logging
from phenotype_network_v0.models.calibration import (
    build_hpo_context,
    calibrate_against_null_pool,
    calibration_config_hash,
    degree_correlations,
    match_null_profiles,
    rank_calibrated_query,
    summarize_phenotype_profiles,
)
from phenotype_network_v0.models.rwr import build_seed_matrix, run_rwr


def _read_ids(path: Path) -> tuple[str, ...]:
    return tuple(sorted(
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ))


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _ids_sha256(values: tuple[str, ...]) -> str:
    payload = chr(10).join(values) + chr(10)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _summary(values: pd.Series) -> dict[str, float | int | None]:
    numeric = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)
    if not len(numeric):
        return {"count": 0, "minimum": None, "mean": None, "median": None, "maximum": None}
    return {
        "count": len(numeric),
        "minimum": float(numeric.min()),
        "mean": float(numeric.mean()),
        "median": float(np.median(numeric)),
        "maximum": float(numeric.max()),
    }


def _raw_score_arrays(
    rankings: pd.DataFrame,
    disease_ids: tuple[str, ...],
    candidate_gene_ids: tuple[str, ...],
) -> dict[str, np.ndarray]:
    candidate_position = {gene: index for index, gene in enumerate(candidate_gene_ids)}
    result = {}
    groups = {str(disease): group for disease, group in rankings.groupby("disease_id", sort=False)}
    if set(groups) != set(disease_ids):
        raise ValueError("RWR rankings do not cover exactly the test diseases")
    for disease_id in disease_ids:
        group = groups[disease_id]
        if len(group) != len(candidate_gene_ids) or group["gene_id"].astype(str).duplicated().any():
            raise ValueError(f"Incomplete or duplicate raw ranking: {disease_id}")
        positions = group["gene_id"].astype(str).map(candidate_position)
        if positions.isna().any():
            raise ValueError(f"Raw ranking contains a gene outside the candidate universe: {disease_id}")
        values = np.empty(len(candidate_gene_ids), dtype=float)
        values[positions.to_numpy(dtype=int)] = group["raw_rwr_score"].to_numpy(dtype=float)
        if not np.isfinite(values).all() or np.any(values < 0):
            raise ValueError(f"Raw RWR scores are invalid: {disease_id}")
        result[disease_id] = values
    return result


def _prepare_profile_audit(train_profiles: pd.DataFrame, test_profiles: pd.DataFrame) -> pd.DataFrame:
    train = train_profiles.copy()
    train["profile_split"] = "train_null_pool"
    test = test_profiles.copy()
    test["profile_split"] = "test_query"
    result = pd.concat([train, test], ignore_index=True)
    for column in (
        "top_level_system_ids",
        "unresolved_depth_term_ids",
        "unresolved_system_term_ids",
    ):
        result[column] = result[column].map(list)
    return result


def _temp_path(destination: Path) -> Path:
    return destination.with_name(destination.name + ".tmp")


def main() -> None:
    """Run matched-null score calibration for one or all folds."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "configs/default.yaml",
    )
    parser.add_argument("--fold", type=int, help="Run one fold only")
    parser.add_argument("--mode", choices=("development", "final"), default="final")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logger = configure_logging(level=args.log_level, logger_name=__name__)
    config = load_config(args.config)
    processed, folds, graphs, outputs = (
        Path(config["paths"][key])
        for key in ("processed", "folds", "graphs", "outputs")
    )
    calibration_settings = config["calibration"]
    null_queries = int(calibration_settings[f"{args.mode}_null_queries"])
    number_of_folds = int(config["number_of_folds"])
    fold_indices = [args.fold] if args.fold is not None else list(range(number_of_folds))
    if any(index < 0 or index >= number_of_folds for index in fold_indices):
        raise ValueError("--fold is outside the configured range")

    phenotype_path = processed / "cardiovascular_disease_phenotypes.parquet"
    hpo_edges_path = processed / "hpo_edges.parquet"
    phenotypes = pd.read_parquet(phenotype_path)
    hpo_edges = pd.read_parquet(hpo_edges_path)
    context = build_hpo_context(hpo_edges)
    candidate_gene_ids = tuple(
        pd.read_parquet(folds / "candidate_gene_universe.parquet")["gene_id"].astype(str)
    )
    candidate_hash = _ids_sha256(candidate_gene_ids)
    phenotype_hash = _file_sha256(phenotype_path)
    hpo_edges_hash = _file_sha256(hpo_edges_path)

    for fold_index in fold_indices:
        fold_dir = folds / f"fold_{fold_index}"
        graph_dir = graphs / f"fold_{fold_index}"
        train_diseases = _read_ids(fold_dir / "train_diseases.txt")
        test_diseases = _read_ids(fold_dir / "test_diseases.txt")
        if set(train_diseases) & set(test_diseases):
            raise ValueError(f"Train/test overlap in fold {fold_index}")

        folder_name = "calibrated" if args.mode == "final" else "calibrated_development"
        output_dir = outputs / "rankings" / f"fold_{fold_index}" / folder_name
        destinations = {
            "rankings": output_dir / "rankings.parquet",
            "matches": output_dir / "null_query_matches.parquet",
            "profiles": output_dir / "profile_audit.parquet",
            "correlations": output_dir / "bias_correlations.parquet",
            "qc": output_dir / "calibration_qc.json",
        }
        existing = [path for path in destinations.values() if path.exists()]
        if existing and not args.overwrite:
            raise FileExistsError(
                "Calibration outputs exist; use --overwrite: " + ", ".join(map(str, existing))
            )
        output_dir.mkdir(parents=True, exist_ok=True)
        temporary = {name: _temp_path(path) for name, path in destinations.items()}
        for path in temporary.values():
            if path.exists():
                path.unlink()

        node_map = pd.read_parquet(graph_dir / "node_map.parquet")
        gene_rows = node_map.loc[node_map["node_type"].eq("gene")]
        if tuple(gene_rows["node_id"].astype(str)) != candidate_gene_ids:
            raise ValueError("Graph gene order differs from candidate universe")
        gene_indices = gene_rows["node_index"].to_numpy(dtype=int)
        transition = load_npz(graph_dir / "transition_matrix.npz")
        ppi = load_npz(graph_dir / "A_gene_gene.npz")
        gene_degree = np.asarray(ppi[gene_indices].getnnz(axis=1)).ravel().astype(np.int64)
        graph_degree = np.asarray(transition[gene_indices].getnnz(axis=1)).ravel().astype(np.int64)

        hpo_ic_path = fold_dir / "train_hpo_ic.parquet"
        hpo_ic = pd.read_parquet(hpo_ic_path)
        train_profiles = summarize_phenotype_profiles(
            train_diseases, phenotypes, hpo_ic, context
        )
        test_profiles = summarize_phenotype_profiles(
            test_diseases, phenotypes, hpo_ic, context
        )
        matches = match_null_profiles(
            test_profiles,
            train_profiles,
            null_queries=null_queries,
            matched_pool_size=int(calibration_settings["matched_pool_size"]),
            temperature=float(calibration_settings["matching_temperature"]),
            seed=int(config["random_seed"]) + fold_index,
        )
        if not set(matches["null_disease_id"].astype(str)) <= set(train_diseases):
            raise ValueError("Null matching used a disease outside the training split")
        match_counts = matches.groupby("disease_id").size()
        if set(match_counts.index.astype(str)) != set(test_diseases) or not (
            match_counts == null_queries
        ).all():
            raise ValueError("Null matching did not produce the configured draws per query")

        rwr_dir = outputs / "rankings" / f"fold_{fold_index}" / "rwr"
        rwr_qc = json.loads((rwr_dir / "rwr_qc.json").read_text(encoding="utf-8"))
        alpha = float(rwr_qc["selected_alpha"])
        if not rwr_qc["converged"]:
            raise ValueError(f"Upstream test RWR is not converged in fold {fold_index}")
        batch_size = int(calibration_settings["rwr_batch_size"])
        if batch_size < 1:
            raise ValueError("calibration.rwr_batch_size must be positive")
        null_pool_scores = np.empty(
            (len(candidate_gene_ids), len(train_diseases)), dtype=float
        )
        null_iterations = []
        null_final_diff = 0.0
        null_mass_error = 0.0
        null_minimum_score = np.inf
        fallback_diseases = []
        for start in range(0, len(train_diseases), batch_size):
            stop = min(start + batch_size, len(train_diseases))
            batch_diseases = train_diseases[start:stop]
            batch_seeds, batch_seed_qc = build_seed_matrix(
                batch_diseases, phenotypes, hpo_ic, node_map
            )
            batch_rwr = run_rwr(
                transition,
                batch_seeds,
                alpha=alpha,
                tolerance=float(config["rwr"]["tolerance"]),
                max_iterations=int(config["rwr"]["max_iterations"]),
            )
            if not batch_rwr.converged:
                raise RuntimeError(
                    f"Null-pool RWR did not converge: fold={fold_index} "
                    f"batch={start}:{stop}"
                )
            null_pool_scores[:, start:stop] = batch_rwr.scores[gene_indices, :]
            null_iterations.append(batch_rwr.iterations)
            null_final_diff = max(
                null_final_diff, float(batch_rwr.final_differences.max())
            )
            null_mass_error = max(
                null_mass_error,
                float(np.max(np.abs(batch_rwr.scores.sum(axis=0) - 1.0))),
            )
            null_minimum_score = min(
                null_minimum_score, float(batch_rwr.scores.min())
            )
            fallback_diseases.extend(
                batch_seed_qc["uniform_seed_fallback_diseases"]
            )
            logger.info(
                "Fold %d null RWR batch %d:%d complete: iterations=%d",
                fold_index,
                start,
                stop,
                batch_rwr.iterations,
            )
            del batch_seeds, batch_rwr
        train_seed_qc = {
            "queries": len(train_diseases),
            "uniform_seed_fallback_diseases": fallback_diseases,
        }

        raw_rankings = pd.read_parquet(
            rwr_dir / "rankings.parquet",
            columns=["disease_id", "gene_id", "raw_rwr_score"],
        )
        raw_scores = _raw_score_arrays(raw_rankings, test_diseases, candidate_gene_ids)
        del raw_rankings
        pool_position = {disease: index for index, disease in enumerate(train_diseases)}
        match_groups = {
            str(disease): group.sort_values("draw_index", kind="stable")
            for disease, group in matches.groupby("disease_id", sort=False)
        }

        effective = {
            "algorithm": "matched_training_profile_bootstrap_v1",
            "fold": fold_index,
            "mode": args.mode,
            "null_queries": null_queries,
            "matched_pool_size": int(calibration_settings["matched_pool_size"]),
            "matching_temperature": float(calibration_settings["matching_temperature"]),
            "zero_variance_tolerance": float(
                calibration_settings["zero_variance_tolerance"]
            ),
            "rwr_batch_size": batch_size,
            "random_seed": int(config["random_seed"]) + fold_index,
            "rwr_config_hash": rwr_qc["config_hash"],
            "candidate_gene_ids_sha256": candidate_hash,
            "train_disease_ids_sha256": _ids_sha256(train_diseases),
            "phenotype_table_sha256": phenotype_hash,
            "hpo_edges_sha256": hpo_edges_hash,
            "train_hpo_ic_sha256": _file_sha256(hpo_ic_path),
        }
        config_hash = calibration_config_hash(effective)

        writer: pq.ParquetWriter | None = None
        correlation_records = []
        zero_variance_rows = 0
        try:
            for query_index, disease_id in enumerate(test_diseases, start=1):
                sampled_indices = (
                    match_groups[disease_id]["null_disease_id"]
                    .astype(str)
                    .map(pool_position)
                    .to_numpy(dtype=int)
                )
                calibrated = calibrate_against_null_pool(
                    raw_scores[disease_id],
                    null_pool_scores,
                    sampled_indices,
                    zero_variance_tolerance=float(
                        calibration_settings["zero_variance_tolerance"]
                    ),
                )
                ranking = rank_calibrated_query(
                    disease_id,
                    candidate_gene_ids,
                    raw_scores[disease_id],
                    calibrated,
                    gene_degree,
                    graph_degree,
                    null_query_count=null_queries,
                    calibration_hash=config_hash,
                )
                zero_count = int(calibrated.zero_variance.sum())
                zero_variance_rows += zero_count
                correlations = degree_correlations(ranking)
                correlations.update({
                    "fold": fold_index,
                    "disease_id": disease_id,
                    "candidate_genes": len(candidate_gene_ids),
                    "zero_null_variance_genes": zero_count,
                })
                correlation_records.append(correlations)
                table = pa.Table.from_pandas(ranking, preserve_index=False)
                if writer is None:
                    writer = pq.ParquetWriter(
                        temporary["rankings"], table.schema, compression="zstd"
                    )
                writer.write_table(table)
                if query_index % 10 == 0 or query_index == len(test_diseases):
                    logger.info(
                        "Fold %d calibrated %d/%d test queries",
                        fold_index,
                        query_index,
                        len(test_diseases),
                    )
        finally:
            if writer is not None:
                writer.close()
        if writer is None:
            raise ValueError("No calibrated rankings were produced")

        correlations = pd.DataFrame.from_records(correlation_records)
        profile_audit = _prepare_profile_audit(train_profiles, test_profiles)
        matches.to_parquet(temporary["matches"], index=False)
        profile_audit.to_parquet(temporary["profiles"], index=False)
        correlations.to_parquet(temporary["correlations"], index=False)

        unresolved_depth_ids = sorted({
            term
            for values in profile_audit["unresolved_depth_term_ids"]
            for term in values
        })
        unresolved_system_ids = sorted({
            term
            for values in profile_audit["unresolved_system_term_ids"]
            for term in values
        })
        unique_nulls = matches.groupby("disease_id")["null_disease_id"].nunique()
        qc: dict[str, Any] = {
            "fold": fold_index,
            "mode": args.mode,
            "algorithm": effective["algorithm"],
            "input_counts": {
                "train_null_pool_diseases": len(train_diseases),
                "test_queries": len(test_diseases),
                "candidate_genes": len(candidate_gene_ids),
            },
            "output_counts": {
                "ranking_rows": len(test_diseases) * len(candidate_gene_ids),
                "null_match_rows": len(matches),
                "profile_audit_rows": len(profile_audit),
                "correlation_rows": len(correlations),
            },
            "removed_records": 0,
            "unresolved_identifiers": {
                "depth_hpo_ids": unresolved_depth_ids,
                "top_level_system_hpo_ids": unresolved_system_ids,
                "depth_profile_term_rows": int(
                    profile_audit["unresolved_depth_term_ids"].map(len).sum()
                ),
                "system_profile_term_rows": int(
                    profile_audit["unresolved_system_term_ids"].map(len).sum()
                ),
            },
            "leakage_checks": {
                "train_test_disjoint": not bool(set(train_diseases) & set(test_diseases)),
                "null_pool_is_train_only": set(matches["null_disease_id"].astype(str))
                <= set(train_diseases),
                "test_labels_used": False,
            },
            "matching": {
                "null_queries_per_test_query": null_queries,
                "matched_pool_size": int(calibration_settings["matched_pool_size"]),
                "matching_temperature": float(calibration_settings["matching_temperature"]),
                "total_distance": _summary(matches["total_match_distance"]),
                "phenotype_count_distance": _summary(
                    matches["phenotype_count_distance"]
                ),
                "depth_distance": _summary(matches["depth_distance"]),
                "ic_distance": _summary(matches["ic_distance"]),
                "system_distance": _summary(matches["system_distance"]),
                "unique_null_profiles_per_query": _summary(unique_nulls),
            },
            "null_pool_rwr": {
                "selected_alpha": alpha,
                "batch_size": batch_size,
                "batches": len(null_iterations),
                "maximum_iterations": max(null_iterations),
                "maximum_final_diff": null_final_diff,
                "maximum_mass_error": null_mass_error,
                "minimum_score": null_minimum_score,
                "converged": True,
                "seed_qc": train_seed_qc,
            },
            "zero_variance": {
                "rows": zero_variance_rows,
                "fraction": zero_variance_rows
                / (len(test_diseases) * len(candidate_gene_ids)),
                "tolerance": float(calibration_settings["zero_variance_tolerance"]),
            },
            "degree_bias": {
                "raw_spearman": _summary(
                    correlations["raw_score_gene_degree_spearman"]
                ),
                "corrected_spearman": _summary(
                    correlations["corrected_score_gene_degree_spearman"]
                ),
                "z_score_spearman": _summary(
                    correlations["z_score_gene_degree_spearman"]
                ),
                "absolute_correlation_reduction": _summary(
                    correlations["absolute_correlation_reduction"]
                ),
            },
            "effective_config": effective,
            "calibration_hash": config_hash,
        }
        temporary["qc"].write_text(
            json.dumps(qc, indent=2, sort_keys=True) + chr(10),
            encoding="utf-8",
        )
        for name in ("rankings", "matches", "profiles", "correlations", "qc"):
            temporary[name].replace(destinations[name])
        logger.info(
            "Fold %d calibration complete: mode=%s nulls=%d rows=%d zero_variance=%d",
            fold_index,
            args.mode,
            null_queries,
            qc["output_counts"]["ranking_rows"],
            zero_variance_rows,
        )


if __name__ == "__main__":
    main()
