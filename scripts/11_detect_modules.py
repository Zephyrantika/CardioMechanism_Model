"""Detect stable phenotype-driven Leiden modules and database enrichments."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix, csr_matrix, load_npz, triu

from phenotype_network_v0.config import load_config
from phenotype_network_v0.evaluation.ablation import (
    ALL_RELATIONS,
    ppi_relation_matrix,
)
from phenotype_network_v0.graph.graph_builder import compose_transition_matrix
from phenotype_network_v0.logging_utils import configure_logging
from phenotype_network_v0.models.modules import (
    build_enrichment_index,
    detect_leiden_modules,
    enrich_module_from_index,
    match_module_stability,
    module_topology,
)
from phenotype_network_v0.models.rwr import build_seed_matrix, run_rwr


def _read_ids(path: Path) -> tuple[str, ...]:
    return tuple(sorted(
        line.strip() for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ))


def _stable_seed(seed: int, *values: object) -> int:
    payload = "|".join(map(str, (seed, *values))).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "little")


def _load_relations(graph_dir: Path) -> dict[str, csr_matrix]:
    return {
        name: load_npz(graph_dir / f"A_{name}.npz").tocsr()
        for name in sorted(ALL_RELATIONS)
    }


def _top_genes(
    scores: np.ndarray,
    gene_indices: np.ndarray,
    candidate_gene_ids: tuple[str, ...],
    count: int,
) -> tuple[str, ...]:
    values = np.asarray(scores[gene_indices], dtype=float)
    numbers = np.asarray(candidate_gene_ids, dtype=np.int64)
    order = np.lexsort((numbers, -values))[:count]
    genes = np.asarray(candidate_gene_ids, dtype=object)
    return tuple(genes[order])


def _delete_seed_terms(
    seeds: np.ndarray,
    rng: np.random.Generator,
    fraction: float,
) -> tuple[np.ndarray, int]:
    perturbed = seeds.copy()
    removed = 0
    for column in range(perturbed.shape[1]):
        nonzero = np.flatnonzero(perturbed[:, column] > 0)
        count = min(max(1, int(np.floor(len(nonzero) * fraction))), len(nonzero) - 1)
        selected = rng.choice(nonzero, size=count, replace=False)
        perturbed[selected, column] = 0.0
        perturbed[:, column] /= perturbed[:, column].sum()
        removed += count
    return perturbed, removed


def _noise_seed_weights(
    seeds: np.ndarray,
    rng: np.random.Generator,
    standard_deviation: float,
) -> np.ndarray:
    perturbed = seeds.copy()
    for column in range(perturbed.shape[1]):
        nonzero = np.flatnonzero(perturbed[:, column] > 0)
        noise = np.maximum(
            0.01,
            rng.normal(loc=1.0, scale=standard_deviation, size=len(nonzero)),
        )
        perturbed[nonzero, column] *= noise
        perturbed[:, column] /= perturbed[:, column].sum()
    return perturbed


def _drop_ppi_edges(
    matrix: csr_matrix,
    rng: np.random.Generator,
    fraction: float,
) -> tuple[csr_matrix, int]:
    upper = triu(matrix, k=1).tocoo()
    remove_count = int(np.floor(upper.nnz * fraction))
    removed = rng.choice(upper.nnz, size=remove_count, replace=False)
    keep = np.ones(upper.nnz, dtype=bool)
    keep[removed] = False
    rows = upper.row[keep]
    columns = upper.col[keep]
    values = upper.data[keep]
    symmetric = coo_matrix(
        (
            np.concatenate([values, values]),
            (np.concatenate([rows, columns]), np.concatenate([columns, rows])),
        ),
        shape=matrix.shape,
    ).tocsr()
    return symmetric, remove_count


def main() -> None:
    """Run base module detection, 50 real propagation perturbations, and enrichment."""
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
    destinations = {
        "modules": outputs / "modules" / "modules.parquet",
        "rejected": outputs / "modules" / "rejected_modules.parquet",
        "members": outputs / "modules" / "module_members.parquet",
        "enrichment": outputs / "modules" / "module_enrichment.parquet",
        "perturbation": outputs / "modules" / "perturbation_qc.parquet",
        "qc": outputs / "modules" / "module_qc.json",
    }
    existing = [path for path in destinations.values() if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(
            "Module outputs exist; use --overwrite: " + ", ".join(map(str, existing))
        )
    for path in destinations.values():
        path.parent.mkdir(parents=True, exist_ok=True)

    random_seed = int(config["random_seed"])
    top_count = int(config["module_detection"]["candidate_gene_count"])
    minimum_size = int(config["module_detection"]["minimum_module_size"])
    maximum_size = int(config["module_detection"]["maximum_module_size"])
    perturbation_runs = int(config["module_detection"]["perturbation_runs"])
    consensus_threshold = float(
        config["module_detection"]["minimum_consensus_frequency"]
    )
    if perturbation_runs != 50:
        raise ValueError("Milestone 11 preregisters exactly 50 perturbation runs")

    candidate_gene_ids = tuple(
        pd.read_parquet(folds / "candidate_gene_universe.parquet")["gene_id"].astype(str)
    )
    universe = set(candidate_gene_ids)
    phenotypes = pd.read_parquet(
        processed / "cardiovascular_disease_phenotypes.parquet"
    )
    ppi700 = pd.read_parquet(processed / "string_gene_edges_700.parquet")
    ppi900 = pd.read_parquet(processed / "string_gene_edges_900.parquet")

    reactome = pd.read_parquet(processed / "reactome_gene_pathway.parquet")[
        ["gene_id", "pathway_id", "pathway_name"]
    ].drop_duplicates()
    reactome_index = build_enrichment_index(
        reactome,
        universe,
        term_column="pathway_id",
        name_column="pathway_name",
    )
    go = pd.read_parquet(processed / "go_gene_annotations.parquet").merge(
        pd.read_parquet(processed / "go_terms.parquet")[["go_id", "go_name"]],
        on="go_id",
        how="inner",
        validate="many_to_one",
    )[["gene_id", "go_id", "go_name"]].drop_duplicates()
    go_index = build_enrichment_index(
        go, universe, term_column="go_id", name_column="go_name"
    )

    module_records: list[dict[str, Any]] = []
    member_records: list[dict[str, Any]] = []
    perturbation_records: list[dict[str, Any]] = []
    module_members: dict[str, tuple[str, ...]] = {}

    for fold in range(int(config["number_of_folds"])):
        fold_dir = folds / f"fold_{fold}"
        graph_dir = graphs / f"fold_{fold}"
        test = _read_ids(fold_dir / "test_diseases.txt")
        node_map = pd.read_parquet(graph_dir / "node_map.parquet")
        gene_rows = node_map.loc[node_map["node_type"].eq("gene")]
        if tuple(gene_rows["node_id"].astype(str)) != candidate_gene_ids:
            raise ValueError(f"Fold {fold} gene order differs from candidate universe")
        gene_indices = gene_rows["node_index"].to_numpy(dtype=int)
        hpo_ic = pd.read_parquet(fold_dir / "train_hpo_ic.parquet")
        seeds, _ = build_seed_matrix(test, phenotypes, hpo_ic, node_map)
        transition = load_npz(graph_dir / "transition_matrix.npz").tocsr()
        rwr_qc = json.loads(
            (outputs / "rankings" / f"fold_{fold}" / "rwr" / "rwr_qc.json")
            .read_text(encoding="utf-8")
        )
        alpha = float(rwr_qc["selected_alpha"])
        rankings = pd.read_parquet(
            outputs / "rankings" / f"fold_{fold}" / "rwr" / "rankings.parquet",
            columns=["disease_id", "gene_id", "raw_rwr_score", "rank"],
        )
        rankings = rankings.loc[rankings["rank"].le(top_count)].copy()
        calibrated = pd.read_parquet(
            outputs / "rankings" / f"fold_{fold}" / "calibrated" / "rankings.parquet",
            columns=["disease_id", "gene_id", "z_score"],
        )
        z_lookup = {
            (str(row.disease_id), str(row.gene_id)): float(row.z_score)
            for row in calibrated.itertuples(index=False)
        }
        score_lookup = {
            (str(row.disease_id), str(row.gene_id)): float(row.raw_rwr_score)
            for row in rankings.itertuples(index=False)
        }
        base_modules: dict[str, list[tuple[str, ...]]] = {}
        for disease_id, group in rankings.groupby("disease_id", sort=True):
            ordered = tuple(
                group.sort_values("rank", kind="stable")["gene_id"].astype(str)
            )
            base_modules[str(disease_id)] = detect_leiden_modules(
                ordered,
                ppi700,
                seed=_stable_seed(random_seed, fold, disease_id, "base"),
                minimum_size=minimum_size,
                maximum_size=maximum_size,
            )
        del calibrated, rankings

        relations = _load_relations(graph_dir)
        string900_matrix = ppi_relation_matrix(ppi900, node_map)
        perturbed: dict[str, list[list[tuple[str, ...]]]] = {
            disease_id: [] for disease_id in test
        }
        for run in range(perturbation_runs):
            rng = np.random.default_rng(_stable_seed(random_seed, fold, run))
            run_seeds = seeds
            run_transition = transition
            removed_count = 0
            if run < 15:
                kind = "delete_10pct_hpo"
                run_seeds, removed_count = _delete_seed_terms(seeds, rng, 0.10)
            elif run < 30:
                kind = "delete_10pct_ppi"
                changed = dict(relations)
                changed["gene_gene"], removed_count = _drop_ppi_edges(
                    relations["gene_gene"], rng, 0.10
                )
                run_transition = compose_transition_matrix(
                    changed, config["relation_budget"]
                )
            elif run < 40:
                kind = "string700_900_switch"
                if run % 2:
                    changed = dict(relations)
                    changed["gene_gene"] = string900_matrix
                    run_transition = compose_transition_matrix(
                        changed, config["relation_budget"]
                    )
            else:
                kind = "phenotype_frequency_noise"
                run_seeds = _noise_seed_weights(seeds, rng, 0.10)
            result = run_rwr(
                run_transition,
                run_seeds,
                alpha=alpha,
                tolerance=float(config["rwr"]["tolerance"]),
                max_iterations=int(config["rwr"]["max_iterations"]),
            )
            if not result.converged:
                raise RuntimeError(
                    f"Module perturbation RWR did not converge: fold={fold} run={run}"
                )
            for column, disease_id in enumerate(test):
                top_genes = _top_genes(
                    result.scores[:, column], gene_indices, candidate_gene_ids, top_count
                )
                modules = detect_leiden_modules(
                    top_genes,
                    ppi700 if not (30 <= run < 40 and run % 2) else ppi900,
                    seed=_stable_seed(random_seed, fold, disease_id, run),
                    minimum_size=minimum_size,
                    maximum_size=maximum_size,
                )
                perturbed[disease_id].append(modules)
            perturbation_records.append({
                "fold": fold,
                "run": run,
                "perturbation_type": kind,
                "seed": _stable_seed(random_seed, fold, run),
                "removed_elements": removed_count,
                "alpha": alpha,
                "iterations": result.iterations,
                "maximum_final_diff": float(result.final_differences.max()),
                "converged": result.converged,
            })
            logger.info(
                "Module perturbation complete: fold=%d run=%d type=%s iterations=%d",
                fold, run, kind, result.iterations,
            )

        for disease_id in test:
            for number, members in enumerate(base_modules[disease_id], start=1):
                module_id = f"{disease_id}:M{number:02d}"
                topology = module_topology(members, ppi700)
                stability, frequencies = match_module_stability(
                    members, perturbed[disease_id]
                )
                mean_z = float(np.mean([
                    z_lookup[(disease_id, gene_id)] for gene_id in members
                ]))
                mean_raw = float(np.mean([
                    score_lookup[(disease_id, gene_id)] for gene_id in members
                ]))
                module_records.append({
                    "module_id": module_id,
                    "disease_id": disease_id,
                    "fold": fold,
                    "module_size": len(members),
                    "mean_z_score": mean_z,
                    "mean_raw_rwr_score": mean_raw,
                    "stability_mean_jaccard": stability,
                    **topology,
                })
                module_members[module_id] = members
                for gene_id in members:
                    member_records.append({
                        "module_id": module_id,
                        "disease_id": disease_id,
                        "fold": fold,
                        "gene_id": gene_id,
                        "raw_rwr_score": score_lookup[(disease_id, gene_id)],
                        "z_score": z_lookup[(disease_id, gene_id)],
                        "consensus_frequency": frequencies[gene_id],
                        "consensus_member": frequencies[gene_id] >= consensus_threshold,
                    })
        logger.info("Module stability complete for fold %d", fold)

    modules = pd.DataFrame.from_records(module_records)
    if modules.empty:
        raise ValueError("No Leiden modules passed the preregistered size limits")
    modules["mean_z_percentile"] = modules.groupby("disease_id")[
        "mean_z_score"
    ].rank(method="average", pct=True)
    modules["module_score"] = (
        0.50 * modules["mean_z_percentile"]
        + 0.25 * modules["internal_density"]
        + 0.25 * modules["stability_mean_jaccard"]
    )
    modules["eligible_module"] = ~modules["single_hub_dominated"]
    rejected_modules = modules.loc[~modules["eligible_module"]].copy()
    modules = modules.loc[modules["eligible_module"]].copy()
    eligible_module_ids = set(modules["module_id"])

    enrichment_records: list[dict[str, Any]] = []
    labels: dict[str, str] = {}
    for row in modules.itertuples(index=False):
        candidates = []
        for source, index in (("Reactome", reactome_index), ("GO", go_index)):
            enrichment = enrich_module_from_index(module_members[row.module_id], index)
            if enrichment.empty:
                continue
            enrichment = enrichment.head(50).copy()
            enrichment["module_id"] = row.module_id
            enrichment["disease_id"] = row.disease_id
            enrichment["fold"] = row.fold
            enrichment["source"] = source
            enrichment["significant_fdr_0_05"] = enrichment["fdr_bh"].le(0.05)
            enrichment_records.extend(enrichment.to_dict("records"))
            significant = enrichment.loc[enrichment["fdr_bh"].le(0.05)]
            if not significant.empty:
                best = significant.iloc[0]
                candidates.append((float(best["fdr_bh"]), source, str(best["term_name"])))
        labels[row.module_id] = min(candidates)[2] if candidates else "UNRESOLVED"
    modules["database_supported_label"] = modules["module_id"].map(labels)
    members = pd.DataFrame.from_records(member_records)
    members = members.loc[members["module_id"].isin(eligible_module_ids)].copy()
    enrichment = pd.DataFrame.from_records(enrichment_records)
    perturbation = pd.DataFrame.from_records(perturbation_records)

    expected_runs = int(config["number_of_folds"]) * perturbation_runs
    if len(perturbation) != expected_runs or not perturbation["converged"].all():
        raise ValueError("Perturbation coverage or convergence is incomplete")
    if not rejected_modules.empty:
        logger.warning(
            "%d single-hub-dominated modules were moved to the rejection audit",
            len(rejected_modules),
        )

    temporary = {
        name: path.with_name(path.name + ".tmp")
        for name, path in destinations.items()
    }
    for path in temporary.values():
        if path.exists():
            path.unlink()
    modules.to_parquet(temporary["modules"], index=False)
    rejected_modules.to_parquet(temporary["rejected"], index=False)
    members.to_parquet(temporary["members"], index=False)
    if enrichment.empty:
        enrichment = pd.DataFrame(columns=[
            "term_id", "term_name", "overlap_count", "module_size",
            "term_gene_count", "universe_size", "p_value", "overlap_gene_ids",
            "fdr_bh", "module_id", "disease_id", "fold", "source",
            "significant_fdr_0_05",
        ])
    enrichment.to_parquet(temporary["enrichment"], index=False)
    perturbation.to_parquet(temporary["perturbation"], index=False)
    qc = {
        "primary_candidate_ranking": "rwr_raw_top_200",
        "calibrated_z_used_for_module_scoring_only": True,
        "test_labels_used": False,
        "folds": int(config["number_of_folds"]),
        "diseases": int(modules["disease_id"].nunique()),
        "candidate_modules": len(modules) + len(rejected_modules),
        "modules": len(modules),
        "eligible_modules": len(modules),
        "rejected_single_hub_modules": len(rejected_modules),
        "module_members": len(members),
        "consensus_members": int(members["consensus_member"].sum()),
        "perturbation_runs_per_fold": perturbation_runs,
        "total_perturbation_runs": len(perturbation),
        "all_perturbations_converged": bool(perturbation["converged"].all()),
        "consensus_threshold": consensus_threshold,
        "module_size_limits": [minimum_size, maximum_size],
        "module_score_formula": "0.50*mean_z_percentile + 0.25*internal_density + 0.25*mean_jaccard",
        "unresolved_module_labels": int(modules["database_supported_label"].eq("UNRESOLVED").sum()),
        "enrichment_rows": len(enrichment),
    }
    temporary["qc"].write_text(
        json.dumps(qc, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    for name, path in destinations.items():
        temporary[name].replace(path)
    logger.info("Milestone 11 modules complete: modules=%d", len(modules))


if __name__ == "__main__":
    main()