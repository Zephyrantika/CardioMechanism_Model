"""Verify the V1 upstream literature-methods manifest (Milestone 2).

Prints the reproducibility gate status for every retained method. Methods
whose unchanged smoke test has not passed are reported as unavailable and may
not be presented as reproduced results.

Usage:
    python scripts/v1/02_verify_upstream.py
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from phenotype_network_v1.baselines.upstream import (
    load_upstream_methods,
    report_upstream_status,
)

LOGGER = logging.getLogger("v1.verify_upstream")


def _configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Path to upstream_methods.yaml (default: configs/v1/upstream_methods.yaml).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Write the JSON status report to this path.",
    )
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)
    _configure_logging(args.log_level)

    root = Path(__file__).resolve().parents[2]
    manifest = args.manifest or root / "configs" / "v1" / "upstream_methods.yaml"
    methods = load_upstream_methods(manifest)
    status = report_upstream_status(methods)

    for method in methods:
        LOGGER.info(
            "%-24s [%s] license=%-24s smoke=%-28s",
            method.name,
            method.klass,
            method.license,
            method.smoke_status,
        )
    LOGGER.info(
        "reproduced=%d unavailable_required=%s",
        status["reproduced_count"],
        status["required_reproduction_unavailable"],
    )

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(status, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        LOGGER.info("Status written to %s", args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
