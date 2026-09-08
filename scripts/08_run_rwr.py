"""Tune and run batched phenotype-seeded RWR for all folds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.sparse import load_npz

from phenotype_network_v0.config import load_config
from phenotype_network_v0.logging_utils import configure_logging
from phenotype_network_v0.models.rwr import (
    build_seed_matrix,
    evaluate_gene_scores,
    rank_rwr_query,
    run_rwr,
    rwr_config_hash,
)


def _read_ids(path: Path) -> tuple[str, ...]:
    return tuple(sorted(line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()))


def main() -> None:
    """Select alpha on validation and produce complete test rankings."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(__file__).resolve().parents[1] / "configs/default.yaml")
    parser.add_argument("--fold", type=int, help="Run one fold only")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()
    logger = configure_logging(level=args.log_level, logger_name=__name__)
    config = load_config(args.config)
    processed, folds, graphs, outputs = (Path(config["paths"][key]) for key in ("processed", "folds", "graphs", "outputs"))
    phenotypes = pd.read_parquet(processed / "cardiovascular_disease_phenotypes.parquet")
    disease_genes = pd.read_parquet(processed / "cardiovascular_disease_genes.parquet")
    known_genes = {str(disease): set(group["gene_id"].astype(str)) for disease, group in disease_genes.groupby("disease_id")}
    candidates = tuple(pd.read_parquet(folds / "candidate_gene_universe.parquet")["gene_id"].astype(str))
    rwr_settings = config["rwr"]
    fold_indices = [args.fold] if args.fold is not None else list(range(int(config["number_of_folds"])))
    if any(index < 0 or index >= int(config["number_of_folds"]) for index in fold_indices):
        raise ValueError("--fold is outside the configured range")
    for fold_index in fold_indices:
        fold_dir, graph_dir = folds / f"fold_{fold_index}", graphs / f"fold_{fold_index}"
        validation, test = _read_ids(fold_dir / "val_diseases.txt"), _read_ids(fold_dir / "test_diseases.txt")
        node_map = pd.read_parquet(graph_dir / "node_map.parquet")
        transition = load_npz(graph_dir / "transition_matrix.npz")
        hpo_ic = pd.read_parquet(fold_dir / "train_hpo_ic.parquet")
        gene_rows = node_map.loc[node_map["node_type"].eq("gene")]
        if tuple(gene_rows["node_id"].astype(str)) != candidates:
            raise ValueError("Graph gene order differs from candidate universe")
        gene_indices = gene_rows["node_index"].to_numpy(dtype=int)
        validation_seeds, validation_seed_qc = build_seed_matrix(validation, phenotypes, hpo_ic, node_map)
        validation_metrics = []
        for alpha in config["rwr"]["alpha_grid"]:
            result = run_rwr(transition, validation_seeds, alpha=float(alpha),
                             tolerance=float(rwr_settings["tolerance"]),
                             max_iterations=int(rwr_settings["max_iterations"]))
            if not result.converged:
                raise RuntimeError(f"Validation RWR did not converge: fold={fold_index} alpha={alpha}")
            metrics = evaluate_gene_scores(result.scores, validation, gene_indices, candidates, known_genes)
            metrics.update({"alpha": float(alpha), "iterations": result.iterations,
                            "maximum_final_diff": float(result.final_differences.max())})
            validation_metrics.append(metrics)
        selected = max(validation_metrics, key=lambda value: (value["mrr"], value["recall_at_10"], -value["alpha"]))
        alpha = float(selected["alpha"])
        test_seeds, test_seed_qc = build_seed_matrix(test, phenotypes, hpo_ic, node_map)
        test_result = run_rwr(transition, test_seeds, alpha=alpha,
                              tolerance=float(rwr_settings["tolerance"]),
                              max_iterations=int(rwr_settings["max_iterations"]))
        if not test_result.converged:
            raise RuntimeError(f"Test RWR did not converge: fold={fold_index}")
        effective = {"fold": fold_index, "alpha": alpha, "tolerance": float(rwr_settings["tolerance"]),
                     "max_iterations": int(rwr_settings["max_iterations"]),
                     "transition_sha256": json.loads((graph_dir / "graph_qc.json").read_text())["transition_sha256"]}
        config_hash = rwr_config_hash(effective)
        output_dir = outputs / "rankings" / f"fold_{fold_index}" / "rwr"
        rankings_path, metrics_path, qc_path = (output_dir / name for name in
                                                ("rankings.parquet", "validation_metrics.json", "rwr_qc.json"))
        existing = [path for path in (rankings_path, metrics_path, qc_path) if path.exists()]
        if existing and not args.overwrite:
            raise FileExistsError("RWR outputs exist; use --overwrite: " + ", ".join(map(str, existing)))
        output_dir.mkdir(parents=True, exist_ok=True)
        temp = rankings_path.with_suffix(".parquet.tmp")
        if temp.exists():
            temp.unlink()
        writer = None
        try:
            for column, disease_id in enumerate(test):
                ranking = rank_rwr_query(
                    disease_id, test_result.scores[gene_indices, column], candidates, alpha=alpha,
                    iterations=test_result.iterations, final_difference=float(test_result.final_differences[column]),
                    config_hash=config_hash,
                )
                table = pa.Table.from_pandas(ranking, preserve_index=False)
                if writer is None:
                    writer = pq.ParquetWriter(temp, table.schema, compression="zstd")
                writer.write_table(table)
        finally:
            if writer is not None:
                writer.close()
        temp.replace(rankings_path)
        metrics_path.write_text(json.dumps({"selected_alpha": alpha, "selection_metric": "mrr",
                                            "results": validation_metrics}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        qc = {"fold": fold_index, "test_queries": len(test), "candidate_genes": len(candidates),
              "ranking_rows": len(test) * len(candidates), "selected_alpha": alpha,
              "iterations": test_result.iterations,
              "maximum_final_diff": float(test_result.final_differences.max()),
              "minimum_score": float(test_result.scores.min()), "maximum_mass_error": float(
                  np.max(np.abs(test_result.scores.sum(axis=0) - 1.0))),
              "validation_seed_qc": validation_seed_qc, "test_seed_qc": test_seed_qc,
              "config_hash": config_hash, "converged": test_result.converged}
        qc_path.write_text(json.dumps(qc, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        logger.info("Fold %d RWR complete: alpha=%.2f iterations=%d", fold_index, alpha, test_result.iterations)


if __name__ == "__main__":
    main()