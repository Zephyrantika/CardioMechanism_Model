"""Clean Reactome pathways for Milestone 4."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from phenotype_network_v0.config import load_config
from phenotype_network_v0.data.reactome_parser import clean_reactome, default_pathway_output_paths, write_pathway_outputs
from phenotype_network_v0.logging_utils import configure_logging


def main() -> None:
    """Run the Reactome cleaning command-line interface."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(__file__).resolve().parents[1] / "configs/default.yaml")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()
    logger = configure_logging(level=args.log_level, logger_name=__name__)
    config = load_config(args.config)
    settings = config["reactome"]
    benchmark_path = Path(config["paths"]["processed"]) / "cardiovascular_disease_genes.parquet"
    benchmark = pd.read_parquet(benchmark_path) if benchmark_path.is_file() else pd.DataFrame()
    reference_genes = set(benchmark.get("gene_id", pd.Series(dtype=str)).astype(str))
    logger.info("Cleaning Reactome v%s", settings["version"])
    result = clean_reactome(settings["gene_pathway_file"], settings["pathway_hierarchy_file"],
                            settings["pathway_metadata_file"], species_name=settings["species_name"],
                            minimum_genes=int(config["pathway"]["minimum_genes"]),
                            maximum_genes=int(config["pathway"]["maximum_genes"]),
                            reactome_version=str(settings["version"]), reference_gene_ids=reference_genes)
    write_pathway_outputs(result, default_pathway_output_paths(config), overwrite=args.overwrite)
    logger.info("Reactome cleaning complete: pathways=%d gene_edges=%d", result.qc["output_counts"]["pathways"], len(result.gene_pathway))


if __name__ == "__main__":
    main()