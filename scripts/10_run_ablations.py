"""Run validation-tuned sparse graph ablations for Milestone 10."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.sparse import load_npz

from phenotype_network_v0.config import load_config
from phenotype_network_v0.data.fold_builder import build_train_phenotype_gene_edges
from phenotype_network_v0.evaluation.ablation import (
    ALL_RELATIONS,
    VARIANT_ACTIVE_RELATIONS,
    hpo_gene_relation_matrices,
    ppi_relation_matrix,
    select_active_relations,
    sparse_matrix_sha256,
)
from phenotype_network_v0.evaluation.ranking_metrics import evaluate_ordered_ranking
from phenotype_network_v0.graph.graph_builder import compose_transition_matrix
from phenotype_network_v0.logging_utils import configure_logging
from phenotype_network_v0.models.rwr import (
    build_seed_matrix,
    evaluate_gene_scores,
    run_rwr,
)


def _read_ids(path: Path) -> tuple[str, ...]:
    return tuple(sorted(
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ))


def _load_relations(graph_dir: Path) -> dict[str, Any]:
    return {
        name: load_npz(graph_dir / f"A_{name}.npz").tocsr()
        for name in sorted(ALL_RELATIONS)
    }


def _ordered_genes(
    scores: np.ndarray,
    candidate_gene_ids: tuple[str, ...],
) -> tuple[str, ...]:
    values = np.asarray(scores, dtype=float)
    numbers = np.asarray([int(value) for value in candidate_gene_ids], dtype=np.int64)
    order = np.lexsort((numbers, -values))
    genes = np.asarray(candidate_gene_ids, dtype=object)
    return tuple(genes[order])


def main() -> None:
    """Run graph relation and seed-weight ablations."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "configs/default.yaml",
    )
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logger = configure_logging(level=args.log_level, logger_name=__name__)
    config = load_config(args.config)
    processed, folds, graphs, outputs = (
        Path(config["paths"][key])
        for key in ("processed", "folds", "graphs", "outputs")
    )
    output_dir = outputs / "metrics" / (
        "ablations_smoke" if args.smoke else "ablations"
    )
    destinations = {
        "per_disease": output_dir / "per_disease_metrics.parquet",
        "validation": output_dir / "validation_selection.parquet",
        "unresolved": output_dir / "unresolved_relation_edges.parquet",
        "qc": output_dir / "ablation_qc.json",
    }
    existing = [path for path in destinations.values() if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(
            "Ablation outputs exist; use --overwrite: " + ", ".join(map(str, existing))
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    phenotypes = pd.read_parquet(
        processed / "cardiovascular_disease_phenotypes.parquet"
    )
    genes = pd.read_parquet(
        processed / "cardiovascular_disease_genes.parquet"
    )
    positives = {
        str(disease): set(group["gene_id"].astype(str))
        for disease, group in genes.groupby("disease_id")
    }
    assignments = pd.read_parquet(folds / "disease_fold_assignments.parquet")
    family = assignments.set_index("disease_id")["disease_family_id"].astype(str).to_dict()
    candidate_gene_ids = tuple(
        pd.read_parquet(folds / "candidate_gene_universe.parquet")["gene_id"].astype(str)
    )
    string900 = pd.read_parquet(
        processed / "string_gene_edges_900.parquet"
    )
    biogrid = pd.read_parquet(
        processed / "biogrid_gene_edges.parquet"
    )
    recall_ks = tuple(int(value) for value in config["evaluation"]["recall_ks"])
    variants = (
        ("hpo_gene_only",)
        if args.smoke
        else tuple(sorted(VARIANT_ACTIVE_RELATIONS))
    )
    fold_indices = (0,) if args.smoke else tuple(range(int(config["number_of_folds"])))
    metric_records: list[dict[str, Any]] = []
    validation_records: list[dict[str, Any]] = []
    unresolved_frames: list[pd.DataFrame] = []
    qc_records = []

    for fold in fold_indices:
        fold_dir = folds / f"fold_{fold}"
        graph_dir = graphs / f"fold_{fold}"
        train = _read_ids(fold_dir / "train_diseases.txt")
        validation = _read_ids(fold_dir / "val_diseases.txt")
        test = _read_ids(fold_dir / "test_diseases.txt")
        if set(train) & set(validation) or set(train) & set(test) or set(validation) & set(test):
            raise ValueError(f"Disease split overlap in fold {fold}")
        hpo_ic = pd.read_parquet(fold_dir / "train_hpo_ic.parquet")
        node_map = pd.read_parquet(graph_dir / "node_map.parquet")
        gene_rows = node_map.loc[node_map["node_type"].eq("gene")]
        if tuple(gene_rows["node_id"].astype(str)) != candidate_gene_ids:
            raise ValueError(f"Fold {fold} graph gene order differs from candidates")
        gene_indices = gene_rows["node_index"].to_numpy(dtype=int)
        base_relations = _load_relations(graph_dir)

        for variant in variants:
            use_frequency = variant != "no_frequency"
            use_ic = variant != "no_ic"
            relations = dict(base_relations)
            edge_qc: dict[str, Any] = {"source": "fold_graph"}
            if variant in {"no_frequency", "no_ic"}:
                rebuilt_edges = build_train_phenotype_gene_edges(
                    set(train),
                    phenotypes,
                    genes,
                    hpo_ic,
                    float(config["folds"]["weight_clip_quantile"]),
                    use_frequency=use_frequency,
                    use_ic=use_ic,
                )
                forward, reverse, unresolved = hpo_gene_relation_matrices(
                    rebuilt_edges, node_map
                )
                if not unresolved.empty:
                    unresolved = unresolved.assign(fold=fold, variant=variant)
                    unresolved_frames.append(unresolved)
                relations["hpo_gene"] = forward
                relations["gene_hpo"] = reverse
                edge_qc = {
                    "source": "rebuilt_train_only",
                    "input_hpo_gene_edges": len(rebuilt_edges),
                    "output_hpo_gene_edges": int(forward.nnz),
                    "removed_hpo_gene_edges": len(unresolved),
                    "unresolved_by_reason": {
                        str(reason): int(count)
                        for reason, count in unresolved["reason"].value_counts().items()
                    },
                    "use_frequency": use_frequency,
                    "use_ic": use_ic,
                }
            if variant in {"biogrid", "string900"}:
                ppi_edges = biogrid if variant == "biogrid" else string900
                relations["gene_gene"] = ppi_relation_matrix(ppi_edges, node_map)
                edge_qc = {
                    "source": (
                        "BioGRID_5.0.260"
                        if variant == "biogrid"
                        else "STRING_v12_threshold_900"
                    ),
                    "gene_gene_nnz": relations["gene_gene"].nnz,
                }
            selected_relations = select_active_relations(
                relations, VARIANT_ACTIVE_RELATIONS[variant]
            )
            transition = compose_transition_matrix(
                selected_relations, config["relation_budget"]
            )
            row_error = float(np.max(
                np.abs(np.asarray(transition.sum(axis=1)).ravel() - 1.0)
            ))
            validation_seeds, validation_seed_qc = build_seed_matrix(
                validation,
                phenotypes,
                hpo_ic,
                node_map,
                use_frequency=use_frequency,
                use_ic=use_ic,
            )
            validation_results = []
            for alpha_value in config["rwr"]["alpha_grid"]:
                result = run_rwr(
                    transition,
                    validation_seeds,
                    alpha=float(alpha_value),
                    tolerance=float(config["rwr"]["tolerance"]),
                    max_iterations=int(config["rwr"]["max_iterations"]),
                )
                if not result.converged:
                    raise RuntimeError(
                        f"Validation ablation RWR did not converge: "
                        f"fold={fold} variant={variant} alpha={alpha_value}"
                    )
                values = evaluate_gene_scores(
                    result.scores,
                    validation,
                    gene_indices,
                    candidate_gene_ids,
                    positives,
                )
                record = {
                    "fold": fold,
                    "variant": variant,
                    "alpha": float(alpha_value),
                    "iterations": result.iterations,
                    "maximum_final_diff": float(result.final_differences.max()),
                    **values,
                }
                validation_results.append(record)
                validation_records.append(record)
            selected = max(
                validation_results,
                key=lambda value: (
                    value["mrr"],
                    value["recall_at_10"],
                    -value["alpha"],
                ),
            )
            alpha = float(selected["alpha"])
            test_seeds, test_seed_qc = build_seed_matrix(
                test,
                phenotypes,
                hpo_ic,
                node_map,
                use_frequency=use_frequency,
                use_ic=use_ic,
            )
            test_result = run_rwr(
                transition,
                test_seeds,
                alpha=alpha,
                tolerance=float(config["rwr"]["tolerance"]),
                max_iterations=int(config["rwr"]["max_iterations"]),
            )
            if not test_result.converged:
                raise RuntimeError(
                    f"Test ablation RWR did not converge: fold={fold} variant={variant}"
                )
            for column, disease_id in enumerate(test):
                ordered = _ordered_genes(
                    test_result.scores[gene_indices, column],
                    candidate_gene_ids,
                )
                metrics = evaluate_ordered_ranking(
                    ordered,
                    positives[disease_id],
                    recall_ks=recall_ks,
                )
                metrics.update({
                    "disease_id": disease_id,
                    "fold": fold,
                    "disease_family_id": family[disease_id],
                    "model": f"ablation_{variant}",
                    "variant": variant,
                    "selected_alpha": alpha,
                })
                metric_records.append(metrics)
            qc_records.append({
                "fold": fold,
                "variant": variant,
                "active_relations": sorted(VARIANT_ACTIVE_RELATIONS[variant]),
                "edge_qc": edge_qc,
                "transition_sha256": sparse_matrix_sha256(transition),
                "transition_nnz": transition.nnz,
                "maximum_row_sum_error": row_error,
                "selected_alpha": alpha,
                "validation_selection": selected,
                "validation_seed_qc": validation_seed_qc,
                "test_seed_qc": test_seed_qc,
                "test_iterations": test_result.iterations,
                "test_maximum_final_diff": float(
                    test_result.final_differences.max()
                ),
                "test_maximum_mass_error": float(np.max(
                    np.abs(test_result.scores.sum(axis=0) - 1.0)
                )),
                "test_minimum_score": float(test_result.scores.min()),
                "train_only_graph_labels": True,
                "validation_labels_used_for_alpha_only": True,
                "test_labels_used_for_tuning": False,
            })
            logger.info(
                "Ablation complete: fold=%d variant=%s alpha=%.2f iterations=%d",
                fold,
                variant,
                alpha,
                test_result.iterations,
            )

    metrics_frame = pd.DataFrame.from_records(metric_records)
    validation_frame = pd.DataFrame.from_records(validation_records)
    unresolved_frame = (
        pd.concat(unresolved_frames, ignore_index=True)
        if unresolved_frames
        else pd.DataFrame(
            columns=["fold", "variant", "hpo_id", "gene_id", "reason"]
        )
    )
    unresolved_frame = unresolved_frame[
        ["fold", "variant", "hpo_id", "gene_id", "reason"]
    ].sort_values(["fold", "variant", "hpo_id", "gene_id"], kind="stable")
    expected_rows = sum(
        len(_read_ids(folds / f"fold_{fold}" / "test_diseases.txt"))
        for fold in fold_indices
    ) * len(variants)
    if len(metrics_frame) != expected_rows:
        raise ValueError(
            f"Expected {expected_rows} ablation metric rows; found {len(metrics_frame)}"
        )
    write_metrics = metrics_frame.copy()
    write_metrics["unevaluable_gene_ids"] = write_metrics[
        "unevaluable_gene_ids"
    ].map(list)
    temporary = {
        name: path.with_name(path.name + ".tmp")
        for name, path in destinations.items()
    }
    for path in temporary.values():
        if path.exists():
            path.unlink()
    write_metrics.to_parquet(temporary["per_disease"], index=False)
    validation_frame.to_parquet(temporary["validation"], index=False)
    unresolved_frame.to_parquet(temporary["unresolved"], index=False)
    qc = {
        "smoke": args.smoke,
        "folds": list(fold_indices),
        "variants": list(variants),
        "per_disease_rows": len(metrics_frame),
        "validation_rows": len(validation_frame),
        "unresolved_relation_edges": len(unresolved_frame),
        "candidate_genes": len(candidate_gene_ids),
        "records": qc_records,
    }
    temporary["qc"].write_text(
        json.dumps(qc, indent=2, sort_keys=True) + chr(10),
        encoding="utf-8",
    )
    for name in ("per_disease", "validation", "unresolved", "qc"):
        temporary[name].replace(destinations[name])


if __name__ == "__main__":
    main()
