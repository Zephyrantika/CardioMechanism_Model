"""Evaluate existing baselines and prepare Milestone 10 comparison outputs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.sparse import load_npz

from phenotype_network_v0.config import load_config
from phenotype_network_v0.evaluation.bootstrap import (
    cluster_bootstrap_confidence_intervals,
    paired_cluster_bootstrap_difference,
)
from phenotype_network_v0.evaluation.ranking_metrics import (
    deterministic_score_order,
    evaluate_ordered_ranking,
    evaluate_positive_ranks,
)
from phenotype_network_v0.logging_utils import configure_logging


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


def _metric_record(
    disease_id: str,
    fold: int,
    family_id: str,
    model: str,
    ordered_genes: tuple[str, ...],
    positive_genes: set[str],
    recall_ks: tuple[int, ...],
) -> dict[str, Any]:
    result = evaluate_ordered_ranking(
        ordered_genes, positive_genes, recall_ks=recall_ks
    )
    result.update({
        "disease_id": disease_id,
        "fold": fold,
        "disease_family_id": family_id,
        "model": model,
    })
    return result


def _random_record(
    disease_id: str,
    fold: int,
    family_id: str,
    candidate_genes: tuple[str, ...],
    positive_genes: set[str],
    recall_ks: tuple[int, ...],
    *,
    repetitions: int,
    seed: int,
) -> dict[str, Any]:
    candidate_set = set(candidate_genes)
    covered = positive_genes & candidate_set
    unresolved = sorted(positive_genes - candidate_set, key=int)
    disease_seed = int.from_bytes(
        hashlib.sha256(disease_id.encode("utf-8")).digest()[:4], "little"
    )
    rng = np.random.default_rng(np.random.SeedSequence([seed, fold, disease_seed]))
    metric_names = [
        *(f"recall_at_{k}" for k in recall_ks),
        "mrr",
        "average_precision",
        "auprc",
        "best_positive_rank",
    ]
    draws = {metric: [] for metric in metric_names}
    for _ in range(repetitions):
        ranks = (
            np.sort(rng.choice(len(candidate_genes), size=len(covered), replace=False) + 1)
            if covered else np.asarray([], dtype=int)
        )
        values = evaluate_positive_ranks(
            ranks,
            candidate_gene_count=len(candidate_genes),
            reference_positive_count=len(positive_genes),
            recall_ks=recall_ks,
        )
        for metric in metric_names:
            if values[metric] is not None:
                draws[metric].append(float(values[metric]))
    record: dict[str, Any] = {
        "disease_id": disease_id,
        "fold": fold,
        "disease_family_id": family_id,
        "model": "random",
        "candidate_gene_count": len(candidate_genes),
        "reference_positive_count": len(positive_genes),
        "covered_positive_count": len(covered),
        "candidate_coverage": (
            len(covered) / len(positive_genes) if positive_genes else None
        ),
        "unevaluable_gene_ids": unresolved,
        "evaluable": bool(covered),
    }
    record.update({
        metric: float(np.mean(values)) if values else None
        for metric, values in draws.items()
    })
    return record


def _summary_rows(
    per_disease: pd.DataFrame,
    metric_columns: tuple[str, ...],
) -> pd.DataFrame:
    records = []
    groupings = [
        ("fold", ["model", "fold"]),
        ("overall", ["model"]),
    ]
    for scope, columns in groupings:
        for keys, group in per_disease.groupby(columns, sort=True):
            key_values = keys if isinstance(keys, tuple) else (keys,)
            identity = dict(zip(columns, key_values, strict=True))
            for metric in (*metric_columns, "candidate_coverage"):
                values = pd.to_numeric(group[metric], errors="coerce").dropna()
                records.append({
                    **identity,
                    "scope": scope,
                    "metric": metric,
                    "diseases": len(group),
                    "evaluable_diseases": int(group["evaluable"].sum()),
                    "macro_mean": float(values.mean()) if len(values) else None,
                    "median": float(values.median()) if len(values) else None,
                    "minimum": float(values.min()) if len(values) else None,
                    "maximum": float(values.max()) if len(values) else None,
                })
    return pd.DataFrame.from_records(records)


def _semantic_records(
    rankings: pd.DataFrame,
    fold: int,
    test_diseases: tuple[str, ...],
    family: dict[str, str],
    positives: dict[str, set[str]],
    recall_ks: tuple[int, ...],
) -> list[dict[str, Any]]:
    records = []
    expected = set(test_diseases)
    if set(rankings["disease_id"].astype(str)) != expected:
        raise ValueError(f"Semantic rankings do not cover fold {fold} test diseases")
    for (disease_id, method), group in rankings.groupby(
        ["disease_id", "aggregation_method"], sort=True
    ):
        ordered = tuple(
            group.sort_values("rank", kind="stable")["gene_id"].astype(str)
        )
        records.append(_metric_record(
            str(disease_id),
            fold,
            family[str(disease_id)],
            f"semantic_{method}",
            ordered,
            positives[str(disease_id)],
            recall_ks,
        ))
    return records


def main() -> None:
    """Evaluate existing rankings without using labels for model construction."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "configs/default.yaml",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logger = configure_logging(level=args.log_level, logger_name=__name__)
    config = load_config(args.config)
    processed, folds, graphs, outputs = (
        Path(config["paths"][key])
        for key in ("processed", "folds", "graphs", "outputs")
    )
    output_dir = outputs / "metrics"
    destinations = {
        "per_disease": output_dir / "per_disease_metrics.parquet",
        "summary": output_dir / "metric_summary.parquet",
        "bootstrap": output_dir / "bootstrap_confidence_intervals.parquet",
        "paired": output_dir / "paired_model_differences.json",
        "qc": output_dir / "evaluation_qc.json",
    }
    existing = [path for path in destinations.values() if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(
            "Evaluation outputs exist; use --overwrite: " + ", ".join(map(str, existing))
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    candidate_genes = tuple(
        pd.read_parquet(folds / "candidate_gene_universe.parquet")["gene_id"].astype(str)
    )
    candidate_set = set(candidate_genes)
    gene_labels = pd.read_parquet(
        processed / "cardiovascular_disease_genes.parquet"
    )
    positives = {
        str(disease): set(group["gene_id"].astype(str))
        for disease, group in gene_labels.groupby("disease_id")
    }
    assignments_path = folds / "disease_fold_assignments.parquet"
    assignments = pd.read_parquet(assignments_path)
    family = assignments.set_index("disease_id")["disease_family_id"].astype(str).to_dict()
    recall_ks = tuple(int(value) for value in config["evaluation"]["recall_ks"])
    random_repetitions = int(config["evaluation"]["random_repetitions"])
    records: list[dict[str, Any]] = []
    input_hashes: dict[str, str] = {
        "candidate_universe": _file_sha256(folds / "candidate_gene_universe.parquet"),
        "disease_gene_labels": _file_sha256(
            processed / "cardiovascular_disease_genes.parquet"
        ),
        "fold_assignments": _file_sha256(assignments_path),
    }

    for fold in range(int(config["number_of_folds"])):
        test_diseases = _read_ids(folds / f"fold_{fold}" / "test_diseases.txt")
        if any(disease not in positives for disease in test_diseases):
            raise ValueError(f"Fold {fold} has a test disease without reference genes")

        semantic_path = outputs / "rankings" / f"fold_{fold}" / "semantic" / "rankings.parquet"
        semantic = pd.read_parquet(
            semantic_path,
            columns=["disease_id", "gene_id", "rank", "aggregation_method"],
        )
        records.extend(_semantic_records(
            semantic, fold, test_diseases, family, positives, recall_ks
        ))
        input_hashes[f"fold_{fold}_semantic"] = _file_sha256(semantic_path)
        del semantic

        rwr_path = outputs / "rankings" / f"fold_{fold}" / "rwr" / "rankings.parquet"
        rwr = pd.read_parquet(
            rwr_path, columns=["disease_id", "gene_id", "rank"]
        )
        if set(rwr["disease_id"].astype(str)) != set(test_diseases):
            raise ValueError(f"RWR rankings do not cover fold {fold} test diseases")
        for disease_id, group in rwr.groupby("disease_id", sort=True):
            ordered = tuple(
                group.sort_values("rank", kind="stable")["gene_id"].astype(str)
            )
            records.append(_metric_record(
                str(disease_id), fold, family[str(disease_id)], "rwr_raw",
                ordered, positives[str(disease_id)], recall_ks,
            ))
        input_hashes[f"fold_{fold}_rwr"] = _file_sha256(rwr_path)
        del rwr

        calibrated_path = (
            outputs / "rankings" / f"fold_{fold}" / "calibrated" / "rankings.parquet"
        )
        calibrated = pd.read_parquet(
            calibrated_path,
            columns=["disease_id", "gene_id", "corrected_rank"],
        )
        if set(calibrated["disease_id"].astype(str)) != set(test_diseases):
            raise ValueError(f"Calibrated rankings do not cover fold {fold} test diseases")
        for disease_id, group in calibrated.groupby("disease_id", sort=True):
            ordered = tuple(
                group.sort_values("corrected_rank", kind="stable")["gene_id"].astype(str)
            )
            records.append(_metric_record(
                str(disease_id), fold, family[str(disease_id)], "rwr_calibrated",
                ordered, positives[str(disease_id)], recall_ks,
            ))
        input_hashes[f"fold_{fold}_calibrated"] = _file_sha256(calibrated_path)
        del calibrated

        node_map = pd.read_parquet(graphs / f"fold_{fold}" / "node_map.parquet")
        gene_rows = node_map.loc[node_map["node_type"].eq("gene")]
        if tuple(gene_rows["node_id"].astype(str)) != candidate_genes:
            raise ValueError(f"Fold {fold} graph gene order differs from candidates")
        ppi = load_npz(graphs / f"fold_{fold}" / "A_gene_gene.npz")
        gene_indices = gene_rows["node_index"].to_numpy(dtype=int)
        degree = np.asarray(ppi[gene_indices].getnnz(axis=1)).ravel()
        degree_order = deterministic_score_order(candidate_genes, degree)
        for disease_id in test_diseases:
            records.append(_metric_record(
                disease_id, fold, family[disease_id], "gene_degree",
                degree_order, positives[disease_id], recall_ks,
            ))
            records.append(_random_record(
                disease_id,
                fold,
                family[disease_id],
                candidate_genes,
                positives[disease_id],
                recall_ks,
                repetitions=random_repetitions,
                seed=int(config["random_seed"]),
            ))
        logger.info("Evaluated existing models for fold %d", fold)

    per_disease = pd.DataFrame.from_records(records)
    ablation_path = output_dir / "ablations" / "per_disease_metrics.parquet"
    ablations = pd.read_parquet(ablation_path)
    expected_ablation_models = {
        f"ablation_{variant}" for variant in (
            "biogrid",
            "hpo_gene_only",
            "hpo_gene_ppi",
            "no_hpo_hierarchy",
            "no_ppi",
            "no_reactome",
            "no_frequency",
            "no_ic",
            "string900",
        )
    }
    if set(ablations["model"]) != expected_ablation_models:
        raise ValueError("Ablation evaluation model set is incomplete")
    for model, group in ablations.groupby("model"):
        if set(group["disease_id"].astype(str)) != set(assignments["disease_id"].astype(str)):
            raise ValueError(f"Ablation model {model} does not cover all diseases")
    per_disease = pd.concat([per_disease, ablations], ignore_index=True)
    input_hashes["ablation_metrics"] = _file_sha256(ablation_path)
    expected_models = {
        "semantic_max",
        "semantic_top5_mean",
        "semantic_weighted_sum",
        "rwr_raw",
        "rwr_calibrated",
        "gene_degree",
        "random",
        *expected_ablation_models,
    }
    if set(per_disease["model"]) != expected_models:
        raise ValueError("Evaluation model set is incomplete")
    expected_rows = len(assignments) * len(expected_models)
    if len(per_disease) != expected_rows:
        raise ValueError(
            f"Expected {expected_rows} per-disease metric rows; found {len(per_disease)}"
        )
    metric_columns = tuple(
        [*(f"recall_at_{k}" for k in recall_ks), "mrr", "average_precision", "auprc"]
    )
    summary = _summary_rows(per_disease, metric_columns)
    bootstrap = cluster_bootstrap_confidence_intervals(
        per_disease,
        metric_columns=(*metric_columns, "candidate_coverage"),
        iterations=int(config["evaluation"]["bootstrap_iterations"]),
        seed=int(config["random_seed"]),
    )
    comparisons = []
    for baseline in sorted(expected_models - {"rwr_raw"}):
        for metric in ("recall_at_10", "mrr", "average_precision", "auprc"):
            comparisons.append(paired_cluster_bootstrap_difference(
                per_disease,
                "rwr_raw",
                baseline,
                metric=metric,
                iterations=int(config["evaluation"]["bootstrap_iterations"]),
                seed=int(config["random_seed"]),
            ))

    fold_lookup = summary.loc[
        summary["scope"].eq("fold")
        & summary["metric"].isin(["recall_at_10", "mrr"])
    ]
    majority_checks = []
    for fold in range(int(config["number_of_folds"])):
        selected = fold_lookup.loc[fold_lookup["fold"].eq(fold)]
        primary = selected.loc[
            selected["model"].eq("rwr_raw")
        ].set_index("metric")["macro_mean"]
        semantic = selected.loc[
            selected["model"].str.startswith("semantic_")
        ].groupby("metric")["macro_mean"].max()
        majority_checks.append({
            "fold": fold,
            "primary_model": "rwr_raw",
            "primary_recall_at_10": float(primary["recall_at_10"]),
            "best_semantic_recall_at_10": float(semantic["recall_at_10"]),
            "primary_mrr": float(primary["mrr"]),
            "best_semantic_mrr": float(semantic["mrr"]),
            "beats_best_semantic_on_recall_or_mrr": bool(
                primary["recall_at_10"] > semantic["recall_at_10"]
                or primary["mrr"] > semantic["mrr"]
            ),
        })

    temporary = {
        name: path.with_name(path.name + ".tmp")
        for name, path in destinations.items()
    }
    for path in temporary.values():
        if path.exists():
            path.unlink()
    write_frame = per_disease.copy()
    write_frame["unevaluable_gene_ids"] = write_frame["unevaluable_gene_ids"].map(list)
    write_frame.to_parquet(temporary["per_disease"], index=False)
    summary.to_parquet(temporary["summary"], index=False)
    bootstrap.to_parquet(temporary["bootstrap"], index=False)
    temporary["paired"].write_text(
        json.dumps(comparisons, indent=2, sort_keys=True) + chr(10),
        encoding="utf-8",
    )
    qc = {
        "models": sorted(expected_models),
        "diseases": len(assignments),
        "families": int(assignments["disease_family_id"].nunique()),
        "per_disease_rows": len(per_disease),
        "candidate_genes": len(candidate_genes),
        "reference_gene_count": int(gene_labels["gene_id"].astype(str).nunique()),
        "candidate_reference_gene_count": int(
            gene_labels.loc[
                gene_labels["gene_id"].astype(str).isin(candidate_set), "gene_id"
            ].astype(str).nunique()
        ),
        "diseases_with_no_covered_positive": int(
            per_disease.loc[per_disease["model"].eq("rwr_raw"), "evaluable"].eq(False).sum()
        ),
        "random_repetitions": random_repetitions,
        "bootstrap_iterations": int(config["evaluation"]["bootstrap_iterations"]),
        "recall_ks": list(recall_ks),
        "majority_fold_checks": majority_checks,
        "majority_fold_pass_count": sum(
            item["beats_best_semantic_on_recall_or_mrr"] for item in majority_checks
        ),
        "input_sha256": input_hashes,
        "test_labels_used_for_evaluation_only": True,
        "test_labels_used_for_training_or_tuning": False,
    }
    temporary["qc"].write_text(
        json.dumps(qc, indent=2, sort_keys=True) + chr(10),
        encoding="utf-8",
    )
    for name in ("per_disease", "summary", "bootstrap", "paired", "qc"):
        temporary[name].replace(destinations[name])
    logger.info(
        "Baseline evaluation complete: diseases=%d models=%d rows=%d",
        len(assignments),
        len(expected_models),
        len(per_disease),
    )


if __name__ == "__main__":
    main()
