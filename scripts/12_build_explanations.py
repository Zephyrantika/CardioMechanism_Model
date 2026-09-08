"""Build constrained evidence paths and verify them by critical-edge deletion."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, load_npz

from phenotype_network_v0.config import load_config
from phenotype_network_v0.evaluation.ablation import ALL_RELATIONS
from phenotype_network_v0.graph.graph_builder import compose_transition_matrix
from phenotype_network_v0.logging_utils import configure_logging
from phenotype_network_v0.models.explanations import (
    enumerate_evidence_paths,
    remove_relation_edge,
)
from phenotype_network_v0.models.rwr import build_seed_matrix, run_rwr

RELATION_SOURCES = {
    "hpo_hpo": "HPO v2026-06-23",
    "hpo_gene": "HPO annotations v2026-06-23; train diseases only",
    "gene_gene": "STRING v12.0 threshold 700",
    "gene_pathway": "Reactome v97",
    "pathway_gene": "Reactome v97",
}


def _read_ids(path: Path) -> tuple[str, ...]:
    return tuple(sorted(
        line.strip() for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ))


def _load_relations(graph_dir: Path) -> dict[str, csr_matrix]:
    return {
        name: load_npz(graph_dir / f"A_{name}.npz").tocsr()
        for name in sorted(ALL_RELATIONS)
    }


def main() -> None:
    """Enumerate allowed paths and repropagate after bottleneck-edge deletion."""
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
    output_dir = outputs / ("paths_smoke" if args.smoke else "paths")
    destinations = {
        "paths": output_dir / "evidence_paths.parquet",
        "deletion": output_dir / "edge_deletion_qc.parquet",
        "unresolved": output_dir / "unresolved_explanation_targets.parquet",
        "qc": output_dir / "explanation_qc.json",
    }
    existing = [path for path in destinations.values() if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(
            "Explanation outputs exist; use --overwrite: " + ", ".join(map(str, existing))
        )
    for path in destinations.values():
        path.parent.mkdir(parents=True, exist_ok=True)

    explanation = config["explanation"]
    maximum_paths = int(explanation["paths_per_gene"])
    target_count = int(explanation["target_gene_count"])
    epsilon = float(explanation["epsilon"])
    minimum_contribution = float(explanation["minimum_relative_contribution"])
    penalties = {
        str(name): float(value)
        for name, value in explanation["relation_penalty"].items()
    }
    phenotypes = pd.read_parquet(
        processed / "cardiovascular_disease_phenotypes.parquet"
    )
    phenotype_lookup = {
        str(disease): tuple(sorted(set(group["hpo_id"].astype(str))))
        for disease, group in phenotypes.groupby("disease_id")
    }
    path_records: list[dict[str, Any]] = []
    deletion_records: list[dict[str, Any]] = []
    unresolved_records: list[dict[str, Any]] = []
    expected_targets = 0
    fold_indices = (0,) if args.smoke else tuple(range(int(config["number_of_folds"])))

    for fold in fold_indices:
        fold_dir = folds / f"fold_{fold}"
        graph_dir = graphs / f"fold_{fold}"
        test = _read_ids(fold_dir / "test_diseases.txt")
        if args.smoke:
            test = test[:1]
        expected_targets += len(test) * target_count
        node_map = pd.read_parquet(graph_dir / "node_map.parquet")
        index = {
            (str(row.node_type), str(row.node_id)): int(row.node_index)
            for row in node_map.itertuples(index=False)
        }
        relations = _load_relations(graph_dir)
        hpo_ic = pd.read_parquet(fold_dir / "train_hpo_ic.parquet")
        seeds, _ = build_seed_matrix(test, phenotypes, hpo_ic, node_map)
        rwr_qc = json.loads(
            (outputs / "rankings" / f"fold_{fold}" / "rwr" / "rwr_qc.json")
            .read_text(encoding="utf-8")
        )
        alpha = float(rwr_qc["selected_alpha"])
        rankings = pd.read_parquet(
            outputs / "rankings" / f"fold_{fold}" / "rwr" / "rankings.parquet",
            columns=["disease_id", "gene_id", "raw_rwr_score", "rank"],
        )
        rankings = rankings.loc[rankings["rank"].le(target_count)].copy()
        for column, disease_id in enumerate(test):
            selected = rankings.loc[rankings["disease_id"].eq(disease_id)].sort_values(
                "rank", kind="stable"
            )
            if len(selected) != target_count:
                raise ValueError(
                    f"Disease {disease_id} lacks {target_count} explanation targets"
                )
            for target in selected.itertuples(index=False):
                paths = enumerate_evidence_paths(
                    phenotype_lookup[disease_id],
                    str(target.gene_id),
                    relations,
                    node_map,
                    penalties,
                    maximum_paths=maximum_paths,
                    epsilon=epsilon,
                )
                if not paths:
                    unresolved_records.append({
                        "disease_id": disease_id,
                        "fold": fold,
                        "target_gene_id": str(target.gene_id),
                        "target_rank": int(target.rank),
                        "reason": "no_allowed_evidence_path",
                    })
                    continue
                deletion_result: dict[str, Any] | None = None
                for path_rank, path in enumerate(paths, start=1):
                    record = {
                        "disease_id": disease_id,
                        "fold": fold,
                        "target_gene_id": str(target.gene_id),
                        "target_rank": int(target.rank),
                        "target_raw_rwr_score": float(target.raw_rwr_score),
                        "path_rank": path_rank,
                        **path,
                        "edge_sources": [
                            RELATION_SOURCES[relation]
                            for relation in path["relations"]
                        ],
                        "causal_interpretation_permitted": False,
                        "critical_relation": None,
                        "relative_contribution": None,
                        "main_explanation": False,
                    }
                    path_records.append(record)
                primary = paths[0]
                edge_costs = [
                    -np.log(max(float(weight), epsilon)) + penalties[relation]
                    for relation, weight in zip(
                        primary["relations"], primary["edge_weights"], strict=True
                    )
                ]
                critical_position = int(np.argmax(edge_costs))
                source_index = int(primary["node_indices"][critical_position])
                target_index = int(primary["node_indices"][critical_position + 1])
                critical_relation = str(primary["relations"][critical_position])
                changed = remove_relation_edge(
                    relations,
                    critical_relation,
                    source_index,
                    target_index,
                )
                changed_transition = compose_transition_matrix(
                    changed, config["relation_budget"]
                )
                result = run_rwr(
                    changed_transition,
                    seeds[:, column],
                    alpha=alpha,
                    tolerance=float(config["rwr"]["tolerance"]),
                    max_iterations=int(config["rwr"]["max_iterations"]),
                )
                if not result.converged:
                    raise RuntimeError(
                        f"Edge deletion RWR failed: disease={disease_id} target={target.gene_id}"
                    )
                target_node = index[("gene", str(target.gene_id))]
                perturbed_score = float(result.scores[target_node, 0])
                absolute_change = float(target.raw_rwr_score) - perturbed_score
                relative_contribution = absolute_change / max(
                    abs(float(target.raw_rwr_score)), epsilon
                )
                deletion_result = {
                    "disease_id": disease_id,
                    "fold": fold,
                    "target_gene_id": str(target.gene_id),
                    "target_rank": int(target.rank),
                    "critical_relation": critical_relation,
                    "critical_source_node_id": primary["node_ids"][critical_position],
                    "critical_target_node_id": primary["node_ids"][critical_position + 1],
                    "critical_edge_weight": primary["edge_weights"][critical_position],
                    "base_score": float(target.raw_rwr_score),
                    "perturbed_score": perturbed_score,
                    "absolute_score_change": absolute_change,
                    "relative_contribution": relative_contribution,
                    "main_explanation": relative_contribution > minimum_contribution,
                    "iterations": result.iterations,
                    "maximum_final_diff": float(result.final_differences.max()),
                    "converged": result.converged,
                }
                deletion_records.append(deletion_result)
                path_records[-len(paths)]["critical_relation"] = critical_relation
                path_records[-len(paths)]["relative_contribution"] = relative_contribution
                path_records[-len(paths)]["main_explanation"] = deletion_result[
                    "main_explanation"
                ]
            logger.info(
                "Explanation deletion complete: fold=%d disease=%s", fold, disease_id
            )

    paths = pd.DataFrame.from_records(path_records)
    deletions = pd.DataFrame.from_records(deletion_records)
    unresolved = pd.DataFrame.from_records(unresolved_records)
    if len(deletions) + len(unresolved) != expected_targets:
        raise ValueError("Explanation target coverage is incomplete")
    if not deletions.empty and not deletions["converged"].all():
        raise ValueError("Not all edge deletion runs converged")

    temporary = {
        name: path.with_name(path.name + ".tmp")
        for name, path in destinations.items()
    }
    for path in temporary.values():
        if path.exists():
            path.unlink()
    paths.to_parquet(temporary["paths"], index=False)
    deletions.to_parquet(temporary["deletion"], index=False)
    if unresolved.empty:
        unresolved = pd.DataFrame(columns=[
            "disease_id", "fold", "target_gene_id", "target_rank", "reason"
        ])
    unresolved.to_parquet(temporary["unresolved"], index=False)
    qc = {
        "smoke": args.smoke,
        "diseases": int(paths["disease_id"].nunique()) if len(paths) else 0,
        "target_genes_per_disease": target_count,
        "expected_targets": expected_targets,
        "targets_with_paths": len(deletions),
        "unresolved_targets": len(unresolved),
        "paths": len(paths),
        "maximum_paths_per_target": maximum_paths,
        "maximum_path_length": int(explanation["maximum_path_length"]),
        "allowed_templates": [
            "HPO-HPO-Gene", "HPO-Gene-Gene", "HPO-Gene-Pathway-Gene",
            "HPO-HPO-Gene-Gene",
        ],
        "all_paths_simple": bool(all(
            len(nodes) == len(set(nodes)) for nodes in paths.get("node_ids", [])
        )),
        "all_deletion_runs_converged": bool(
            deletions["converged"].all() if len(deletions) else True
        ),
        "main_explanations": int(deletions.get("main_explanation", pd.Series(dtype=bool)).sum()),
        "minimum_relative_contribution": minimum_contribution,
        "test_labels_used": False,
        "causal_claims_permitted": False,
    }
    temporary["qc"].write_text(
        json.dumps(qc, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    for name, path in destinations.items():
        temporary[name].replace(path)


if __name__ == "__main__":
    main()