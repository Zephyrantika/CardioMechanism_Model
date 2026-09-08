"""Milestone 0 environment check for V1.

Usage:
    python scripts/v1/00_check_environment.py --device cpu
    python scripts/v1/00_check_environment.py --device cuda   # on the GPU server

Writes a structured JSON report under ``outputs/v1/`` and exits non-zero when
the requested device is not usable. CPU checks succeed even when the optional
``deep`` dependency group is not installed yet; CUDA checks require PyTorch
with a usable CUDA device.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from phenotype_network_v1 import __version__
from phenotype_network_v1.checkpoint import git_commit
from phenotype_network_v1.environment import EnvironmentReport, check_environment

LOGGER = logging.getLogger("v1.environment_check")


def _configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )


def _find_repo_root(start: Path) -> Path:
    for candidate in (start, *start.parents):
        if (candidate / ".git").exists():
            return candidate
    return start


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for the JSON report (default: outputs/v1).",
    )
    args = parser.parse_args(argv)
    _configure_logging(args.log_level)

    root = Path(__file__).resolve().parents[2]
    report = check_environment(device=args.device)

    summary = {
        "package_version": __version__,
        "git_commit": git_commit(_find_repo_root(root)),
        **report.to_dict(),
    }
    output_dir = args.output_dir or (root / "outputs" / "v1")
    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / f"environment_report_{args.device}.json"
    temporary = destination.with_name(destination.name + ".tmp")
    temporary.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(destination)

    LOGGER.info("V1 package version: %s", __version__)
    LOGGER.info("Python %s (%s) on %s", report.python, report.implementation, report.platform)
    LOGGER.info("PyTorch available: %s (%s)", report.torch_available, report.torch_version)
    LOGGER.info("PyG available: %s (%s)", report.pyg_available, report.pyg_version)
    LOGGER.info("CUDA: %s", report.cuda)
    LOGGER.info("cuDNN version: %s", report.cudnn_version)
    LOGGER.info("Deterministic settings: %s", report.deterministic)
    LOGGER.info("Device check (%s): %s", args.device, "OK" if report.ok else "NOT AVAILABLE")
    LOGGER.info("Report written to %s", destination)
    return 0 if report.ok else 1


if __name__ == "__main__":
    sys.exit(main())
