"""Build fold-specific V1 learning data (Milestone 1).

For each frozen V0 fold (default: all of 0-4) this script:

1. writes ``data/processed/v1/fold_N/query_instances.parquet``;
2. writes ``data/graphs/v1/fold_N/heterodata.pt`` and ``graph_manifest.json``;
3. writes and checks ``data/graphs/v1/fold_N/leakage_report.json``.

The script exits non-zero when any fold leakage report fails, so a failed
milestone stops immediately (no later milestones are touched).

Usage:
    python scripts/v1/01_build_learning_data.py --fold all
    python scripts/v1/01_build_learning_data.py --fold 0
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from phenotype_network_v1.checkpoint import git_commit
from phenotype_network_v1.data.leakage import run_fold_leakage
from phenotype_network_v1.data.pyg_graph import write_graph_artifacts
from phenotype_network_v1.data.query_dataset import (
    build_fold_queries,
    write_query_instances,
)

LOGGER = logging.getLogger("v1.build_learning_data")
NUMBER_OF_FOLDS = 5


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


def build_fold(fold: int, root: Path) -> bool:
    processed_dir = root / "data" / "processed"
    folds_dir = root / "data" / "folds"
    assignments_path = folds_dir / "disease_fold_assignments.parquet"
    graph_source_dir = root / "data" / "graphs" / f"fold_{fold}"
    v1_processed = root / "data" / "processed" / "v1" / f"fold_{fold}"
    v1_graphs = root / "data" / "graphs" / "v1" / f"fold_{fold}"

    queries = build_fold_queries(
        fold,
        processed_dir=processed_dir,
        folds_dir=folds_dir,
        assignments_path=assignments_path,
    )
    queries_path = v1_processed / "query_instances.parquet"
    write_query_instances(queries, queries_path)
    LOGGER.info("fold %d query instances written to %s", fold, queries_path)

    commit = git_commit(root)
    manifest = write_graph_artifacts(
        fold,
        graph_source_dir,
        v1_graphs,
        git_commit=commit,
    )
    LOGGER.info(
        "fold %d graph manifest: nodes=%d serialization=%s",
        fold,
        manifest["total_nodes"],
        manifest["serialization_sha256"][:12],
    )

    report_path = v1_graphs / "leakage_report.json"
    report = run_fold_leakage(
        fold,
        processed_dir=processed_dir,
        folds_dir=folds_dir,
        queries_path=queries_path,
        manifest_path=v1_graphs / "graph_manifest.json",
        report_path=report_path,
    )
    return report["status"] == "PASS"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fold",
        default="all",
        help="Fold index 0-4, or 'all' (default).",
    )
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)
    _configure_logging(args.log_level)

    root = Path(__file__).resolve().parents[2]
    if args.fold == "all":
        folds = list(range(NUMBER_OF_FOLDS))
    else:
        folds = [int(args.fold)]
        if any(fold < 0 or fold >= NUMBER_OF_FOLDS for fold in folds):
            parser.error(f"--fold must be 0-{NUMBER_OF_FOLDS - 1} or 'all'")

    all_passed = True
    for fold in folds:
        passed = build_fold(fold, root)
        all_passed = all_passed and passed
        LOGGER.info("fold %d leakage: %s", fold, "PASS" if passed else "FAIL")

    if not all_passed:
        LOGGER.error("One or more fold leakage reports FAILED; stopping.")
        return 1
    LOGGER.info("All folds passed leakage checks; learning data is ready.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
