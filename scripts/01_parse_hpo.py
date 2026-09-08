"""Parse HPO OBO data into normalized Milestone 1 artifacts."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from phenotype_network_v0.config import load_config
from phenotype_network_v0.data.hpo_parser import (
    default_hpo_output_paths,
    parse_hpo_ontology,
    write_hpo_outputs,
)
from phenotype_network_v0.logging_utils import configure_logging


def main() -> None:
    """Run the HPO parser command-line interface."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "configs/default.yaml",
    )
    parser.add_argument("--input", type=Path, help="Override the configured HPO OBO file")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logger = configure_logging(level=args.log_level, logger_name=__name__)
    config = load_config(args.config)
    input_path = args.input or config["hpo"]["ontology_file"]
    outputs = default_hpo_output_paths(config)
    logger.info("Parsing HPO ontology: %s", input_path)
    result = parse_hpo_ontology(Path(input_path))
    write_hpo_outputs(result, outputs, overwrite=args.overwrite)
    logger.info(
        "HPO parsing complete: nodes=%d edges=%d replaced=%d unresolved=%d",
        len(result.nodes),
        len(result.edges),
        len(result.obsolete_mapping),
        len(result.unresolved_terms),
    )


if __name__ == "__main__":
    main()

