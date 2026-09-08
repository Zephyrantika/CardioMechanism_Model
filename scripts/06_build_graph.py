"""Build sparse heterogeneous graphs for all disease folds."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from phenotype_network_v0.config import load_config
from phenotype_network_v0.graph.graph_builder import build_heterogeneous_graph, write_graph_outputs
from phenotype_network_v0.logging_utils import configure_logging


def main() -> None:
    """Run Milestone 6 graph construction."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(__file__).resolve().parents[1] / "configs/default.yaml")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()
    logger = configure_logging(level=args.log_level, logger_name=__name__)
    config = load_config(args.config)
    processed, folds, graphs = (Path(config["paths"][key]) for key in ("processed", "folds", "graphs"))
    common = {
        "hpo_nodes": pd.read_parquet(processed / "hpo_nodes.parquet"),
        "hpo_edges": pd.read_parquet(processed / "hpo_edges.parquet"),
        "candidates": pd.read_parquet(folds / "candidate_gene_universe.parquet"),
        "string_edges": pd.read_parquet(processed / "string_gene_edges_700.parquet"),
        "reactome_gene_pathway": pd.read_parquet(processed / "reactome_gene_pathway.parquet"),
        "pathways": pd.read_parquet(processed / "reactome_pathways.parquet"),
        "pathway_hierarchy": pd.read_parquet(processed / "reactome_pathway_hierarchy.parquet"),
        "relation_budget": config["relation_budget"],
    }
    for index in range(int(config["number_of_folds"])):
        logger.info("Building graph for fold %d", index)
        result = build_heterogeneous_graph(
            hpo_gene_edges=pd.read_parquet(folds / f"fold_{index}" / "phenotype_gene_edges.parquet"),
            **common,
        )
        write_graph_outputs(result, graphs / f"fold_{index}", overwrite=args.overwrite)
        logger.info("Fold %d graph complete: nodes=%d transition_nnz=%d", index, len(result.node_map), result.transition.nnz)


if __name__ == "__main__":
    main()