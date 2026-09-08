"""Build the Milestone 2 cardiovascular disease benchmark."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from phenotype_network_v0.config import load_config
from phenotype_network_v0.data.disease_benchmark import (
    build_cardiovascular_benchmark,
    default_benchmark_output_paths,
    write_benchmark_outputs,
)
from phenotype_network_v0.logging_utils import configure_logging


def main() -> None:
    """Run the cardiovascular benchmark command-line interface."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(__file__).resolve().parents[1] / "configs/default.yaml")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()
    logger = configure_logging(level=args.log_level, logger_name=__name__)
    config = load_config(args.config)
    benchmark = config["benchmark"]
    logger.info("Building cardiovascular disease benchmark")
    result = build_cardiovascular_benchmark(
        config["hpo"]["annotation_file"], config["hpo"]["gene_disease_file"],
        config["mondo"]["ontology_file"], Path(config["paths"]["processed"]) / "hpo_nodes.parquet",
        Path(config["paths"]["interim"]) / "obsolete_hpo_mapping.parquet",
        minimum_phenotypes=int(benchmark["minimum_phenotypes_per_disease"]),
        minimum_genes=int(benchmark["minimum_genes_per_disease"]),
    )
    write_benchmark_outputs(result, default_benchmark_output_paths(config), overwrite=args.overwrite)
    logger.info("Benchmark complete: diseases=%d phenotypes=%d genes=%d", len(result.samples), len(result.phenotypes), len(result.genes))


if __name__ == "__main__":
    main()