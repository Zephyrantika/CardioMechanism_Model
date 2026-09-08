"""Create MONDO-family-disjoint five-fold experiments."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import pronto

from phenotype_network_v0.config import load_config
from phenotype_network_v0.data.fold_builder import assign_mondo_family_anchors, make_disease_folds, write_fold_outputs
from phenotype_network_v0.logging_utils import configure_logging


def main() -> None:
    """Run Milestone 5 fold construction."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(__file__).resolve().parents[1] / "configs/default.yaml")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()
    logger = configure_logging(level=args.log_level, logger_name=__name__)
    config = load_config(args.config)
    processed = Path(config["paths"]["processed"])
    samples = pd.read_parquet(processed / "cardiovascular_disease_samples.parquet")
    ontology = pronto.Ontology(config["mondo"]["ontology_file"], encoding="utf-8")
    assignments = assign_mondo_family_anchors(set(samples["disease_id"]), ontology,
                                              anchor_depth=int(config["folds"]["family_anchor_depth"]))
    disease_ids = set(samples["disease_id"])
    parent_child_pairs = {
        (parent.id, disease_id)
        for disease_id in disease_ids
        for parent in ontology[disease_id].superclasses(with_self=False)
        if parent.id in disease_ids
    }
    logger.info("Building %d disease-family folds", config["number_of_folds"])
    result = make_disease_folds(
        samples,
        pd.read_parquet(processed / "cardiovascular_disease_phenotypes.parquet"),
        pd.read_parquet(processed / "cardiovascular_disease_genes.parquet"),
        pd.read_parquet(processed / "hpo_nodes.parquet"),
        pd.read_parquet(processed / "hpo_edges.parquet"),
        pd.read_parquet(processed / "string_gene_edges_700.parquet"),
        pd.read_parquet(processed / "reactome_gene_pathway.parquet"),
        assignments,
        number_of_folds=int(config["number_of_folds"]), seed=int(config["random_seed"]),
        weight_clip_quantile=float(config["folds"]["weight_clip_quantile"]),
        parent_child_pairs=parent_child_pairs,
    )
    write_fold_outputs(result, config["paths"]["folds"], overwrite=args.overwrite)
    logger.info("Fold construction complete: families=%d fold_sizes=%s", result.qc["families"], result.qc["fold_disease_counts"])


if __name__ == "__main__":
    main()