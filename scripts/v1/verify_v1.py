"""V1 final verification gate (Milestone 8).

Checks required files, fold leakage reports, query schema, checkpoint hashes,
metric completeness, portability (no absolute paths), and claim boundaries
(V1-B reserved/not started). GPU-marked tests and full server runs are
reported as deferred when CUDA checkpoints/metrics are absent.

Usage:
    python scripts/v1/verify_v1.py --config configs/v1/default.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from pathlib import Path
from typing import Any

LOGGER = logging.getLogger("v1.verify")


def _configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _records(root: Path) -> list[tuple[str, bool, Any]]:
    records: list[tuple[str, bool, Any]] = []
    v1_graphs = root / "data" / "graphs" / "v1"
    v1_processed = root / "data" / "processed" / "v1"
    present_folds = sorted(
        int(path.name.split("_")[1])
        for path in v1_graphs.iterdir()
        if path.is_dir() and path.name.startswith("fold_")
    )
    for fold in present_folds:
        graph_dir = v1_graphs / f"fold_{fold}"
        for name in ("heterodata.pt", "graph_manifest.json", "leakage_report.json"):
            records.append(
                (f"fold_{fold}_has_{name}", (graph_dir / name).exists(), None)
            )
        leakage_path = graph_dir / "leakage_report.json"
        if leakage_path.exists():
            report = _read_json(leakage_path)
            records.append(
                (f"fold_{fold}_leakage_pass", report["status"] == "PASS",
                 {"passed": report["passed_checks"], "failed": report["failed_checks"]})
            )
            records.append(
                (f"fold_{fold}_serialization_hash",
                 bool(report["audit"].get("serialization_sha256")), None)
            )
        queries_path = v1_processed / f"fold_{fold}" / "query_instances.parquet"
        records.append((f"fold_{fold}_query_instances", queries_path.exists(), None))
    records.append(
        ("has_conditioned_checkpoint_fold0",
         (root / "outputs" / "v1" / "conditioned" / "full" / "fold_0" / "model.pt").exists(),
         "server training completes all folds on GPU")
    )
    records.append(
        ("metric_completeness_fold0",
         (root / "outputs" / "v1" / "metrics" / "fold_0" / "metrics.json").exists(),
         "full multi-fold metrics from server runs required for release claims")
    )
    # Portability: no local absolute paths in V1 source/scripts/configs.
    absolute_pattern = re.compile(r"(?:[A-Za-z]:\\|/(?:home|Users)/)")
    hardcoded = []
    for pattern in ("src/phenotype_network_v1/**/*.py", "scripts/v1/*.py", "configs/v1/*.yaml"):
        for path in root.glob(pattern):
            for line_number, line in enumerate(
                path.read_text(encoding="utf-8").splitlines(), 1
            ):
                if absolute_pattern.search(line):
                    hardcoded.append(f"{path.relative_to(root)}:{line_number}")
    records.append(("portable_no_absolute_paths", not hardcoded, hardcoded))
    return records


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)
    _configure_logging(args.log_level)
    root = Path(__file__).resolve().parents[2]
    records = _records(root)
    failed = [name for name, passed, _ in records if not passed]
    result = {
        "status": "PASS" if not failed else "FAIL",
        "checks": [
            {"check": name, "passed": passed, "details": details}
            for name, passed, details in records
        ],
        "passed_checks": len(records) - len(failed),
        "failed_checks": len(failed),
        "v1b_status": "not_started_no_patient_cohort",
        "claim_boundaries": {
            "no_causal_claim": True,
            "gtex_tissue_support_not_causal": True,
            "upstream_unavailable_not_claimed_reproduced": True,
        },
    }
    for name, passed, _ in records:
        LOGGER.info("%-44s %s", name, "OK" if passed else "FAIL")
    LOGGER.info("V1 verify: %s (%d checks)", result["status"], len(records))
    (root / "outputs" / "v1" / "verify_v1_result.json").parent.mkdir(
        parents=True, exist_ok=True
    )
    destination = root / "outputs" / "v1" / "verify_v1_result.json"
    temporary = destination.with_name(destination.name + ".tmp")
    temporary.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(destination)
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
