"""Clean STRING v12.0 into gene-level PPI networks."""

from __future__ import annotations

import argparse
from pathlib import Path

from phenotype_network_v0.config import load_config
import pandas as pd

from phenotype_network_v0.data.disease_benchmark import CORE_DISEASES
from phenotype_network_v0.data.string_parser import clean_string_ppi, default_ppi_output_paths, write_ppi_outputs
from phenotype_network_v0.logging_utils import configure_logging


def main() -> None:
    """Run the STRING PPI cleaning command-line interface."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(__file__).resolve().parents[1] / "configs/default.yaml")
    parser.add_argument("--chunk-size", type=int, default=500_000)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()
    logger = configure_logging(level=args.log_level, logger_name=__name__)
    config = load_config(args.config)
    settings = config["string"]
    logger.info("Cleaning STRING PPI with chunk_size=%d", args.chunk_size)
    benchmark_path = Path(config["paths"]["processed"]) / "cardiovascular_disease_genes.parquet"
    benchmark_genes = pd.read_parquet(benchmark_path) if benchmark_path.is_file() else pd.DataFrame()
    reference_gene_ids = set(benchmark_genes.get("gene_id", pd.Series(dtype=str)).astype(str))
    core_gene_ids = {
        disease_id: set(benchmark_genes.loc[benchmark_genes["disease_id"].eq(disease_id), "gene_id"].astype(str))
        for disease_id in CORE_DISEASES
    } if not benchmark_genes.empty else {disease_id: set() for disease_id in CORE_DISEASES}
    result = clean_string_ppi(settings["links_file"], settings["aliases_file"],
                              taxonomy_id=int(settings["species_taxonomy_id"]),
                              thresholds=tuple(settings["thresholds"]), chunk_size=args.chunk_size,
                              string_version=str(settings["version"]), primary_threshold=700,
                              reference_gene_ids=reference_gene_ids, core_gene_ids=core_gene_ids)
    write_ppi_outputs(result, default_ppi_output_paths(config), overwrite=args.overwrite)
    logger.info("STRING cleaning complete: edge_counts=%s", result.qc["threshold_edge_counts"])


if __name__ == "__main__":
    main()