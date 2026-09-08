"""Verify end-to-end V0 acceptance and server portability invariants."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import pandas as pd

from phenotype_network_v0.config import load_config
from phenotype_network_v0.logging_utils import configure_logging


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    """Run cross-milestone acceptance checks without changing model outputs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "configs/default.yaml",
    )
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()
    logger = configure_logging(level=args.log_level, logger_name=__name__)
    config = load_config(args.config)
    root = Path(__file__).resolve().parents[1]
    processed, folds, graphs, outputs = (
        Path(config["paths"][key])
        for key in ("processed", "folds", "graphs", "outputs")
    )
    records: list[dict[str, Any]] = []

    def check(name: str, passed: bool, details: Any = None) -> None:
        records.append({"check": name, "passed": bool(passed), "details": details})

    required = [
        processed / "hpo_nodes.parquet",
        processed / "cardiovascular_disease_samples.parquet",
        processed / "string_gene_edges_700.parquet",
        processed / "reactome_gene_pathway.parquet",
        processed / "biogrid_gene_edges.parquet",
        processed / "gwas_gene_support.parquet",
        processed / "gtex_cardiovascular_expression.parquet",
        processed / "go_gene_annotations.parquet",
        outputs / "metrics" / "evaluation_qc.json",
        outputs / "modules" / "module_qc.json",
        outputs / "paths" / "explanation_qc.json",
        outputs / "cases" / "case_qc.json",
        root / "requirements-lock.txt",
        root / "docs" / "server_migration.md",
    ]
    check(
        "required_files_exist",
        all(path.exists() for path in required),
        [str(path) for path in required if not path.exists()],
    )

    leakage_failures = []
    for fold in range(int(config["number_of_folds"])):
        report_path = folds / f"fold_{fold}" / "leakage_report.json"
        if not report_path.exists():
            leakage_failures.append({"fold": fold, "reason": "missing_report"})
            continue
        report = _read_json(report_path)
        failed = [name for name, value in report["checks"].items() if not value]
        if failed:
            leakage_failures.append({"fold": fold, "failed_checks": failed})
        graph_qc = _read_json(graphs / f"fold_{fold}" / "graph_qc.json")
        if graph_qc.get("maximum_row_sum_error", 1.0) > 1e-10:
            leakage_failures.append({"fold": fold, "reason": "transition_not_normalized"})
    check("all_fold_leakage_and_transition_checks", not leakage_failures, leakage_failures)

    evaluation = _read_json(outputs / "metrics" / "evaluation_qc.json")
    check("m10_majority_fold_criterion", evaluation["majority_fold_pass_count"] >= 3,
          evaluation["majority_fold_pass_count"])
    check("m10_test_labels_evaluation_only",
          evaluation["test_labels_used_for_evaluation_only"]
          and not evaluation["test_labels_used_for_training_or_tuning"])
    external = _read_json(outputs / "qc" / "external_validation_qc.json")
    check("external_resources_not_in_training",
          not external["entered_training_graph"]
          and not external["used_for_hyperparameter_selection"]
          and not external["expanded_candidate_universe"])

    module_qc = _read_json(outputs / "modules" / "module_qc.json")
    modules = pd.read_parquet(outputs / "modules" / "modules.parquet")
    perturbations = pd.read_parquet(outputs / "modules" / "perturbation_qc.parquet")
    check("m11_perturbations_complete",
          len(perturbations) == 250 and perturbations["converged"].all())
    check("m11_module_constraints",
          modules["module_size"].between(5, 100).all()
          and not modules["single_hub_dominated"].any()
          and modules["disease_id"].nunique() == 451,
          {"modules": len(modules), "diseases": int(modules["disease_id"].nunique())})
    check("m11_test_labels_unused", not module_qc["test_labels_used"])

    explanation = _read_json(outputs / "paths" / "explanation_qc.json")
    check("m12_target_coverage",
          explanation["targets_with_paths"] + explanation["unresolved_targets"]
          == explanation["expected_targets"])
    check("m12_paths_simple_and_converged",
          explanation["all_paths_simple"] and explanation["all_deletion_runs_converged"])
    check("m12_no_causal_claim", not explanation["causal_claims_permitted"])

    cases = _read_json(outputs / "cases" / "case_qc.json")
    check("m13_cases_complete",
          cases["preregistered_cases"] == 3 and cases["clinical_bridge_rows"] == 15)
    check("m13_no_manual_or_patient_training_data",
          not cases["manual_hpo_or_gene_labels_added"]
          and not cases["patient_level_values_used"]
          and not cases["clinical_features_entered_training_graph"])

    unix_home_pattern = "/" + r"(?:home|Users)/"
    absolute_pattern = re.compile(r"(?:[A-Za-z]:\\|" + unix_home_pattern + r")")
    portable_files = [
        *root.joinpath("src").rglob("*.py"),
        *root.joinpath("scripts").rglob("*.py"),
        *root.joinpath("configs").rglob("*.yaml"),
    ]
    hardcoded = []
    for path in portable_files:
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if absolute_pattern.search(line):
                hardcoded.append(f"{path.relative_to(root)}:{line_number}")
    check("portable_no_absolute_paths", not hardcoded, hardcoded)
    lock = (root / "requirements-lock.txt").read_text(encoding="utf-8")
    check("portable_dependency_lock",
          "file://" not in lock and "-e " not in lock and not absolute_pattern.search(lock))

    failed = [record for record in records if not record["passed"]]
    result = {
        "status": "PASS" if not failed else "FAIL",
        "checks": records,
        "passed_checks": len(records) - len(failed),
        "failed_checks": len(failed),
        "v0_interpretable_cpu_baseline": True,
        "deep_learning_dependencies": False,
        "gpu_required": False,
    }
    destination = outputs / "qc" / "v0_acceptance.json"
    temporary = destination.with_name(destination.name + ".tmp")
    temporary.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(destination)
    if failed:
        raise RuntimeError(
            "V0 acceptance failed: " + ", ".join(record["check"] for record in failed)
        )
    logger.info("V0 acceptance passed: checks=%d", len(records))


if __name__ == "__main__":
    main()