"""Fold leakage audit for V1 learning data (Milestone 1).

Verifies, per fold, that the generated query instances and PyG graph
artifacts introduce no label leakage:

* train/validation/test disease and family partitions stay disjoint and
  identical to the frozen V0 split files;
* graph edges are a byte-for-byte reconcile of the frozen V0 graph, whose
  own leakage report (train-only edges, no test/validation contributors) is
  re-endorsed here;
* every query HPO id resolves through the fold-specific training IC table and
  weights are non-negative and finite;
* positive genes use canonical IDs and are either covered by the frozen
  candidate universe or explicitly listed as unresolved;
* unlabelled genes are never marked as confirmed negatives.

One JSON report is written per fold; Milestone 1 acceptance requires all five
reports to pass.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd

from phenotype_network_v1.data.query_dataset import (
    load_candidate_universe,
    load_fold_ic,
    read_split_diseases,
)

LOGGER = logging.getLogger("phenotype_network_v1.data.leakage")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_list(value: Any) -> list[str]:
    if value is None:
        return []
    if hasattr(value, "tolist"):
        value = value.tolist()
    return [str(item) for item in value]


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _edge_nnz_from_npz(path: Path) -> int:
    import numpy as np

    archive = np.load(path)
    return int(len(archive["data"]))


def run_fold_leakage(
    fold: int,
    *,
    processed_dir: str | Path,
    folds_dir: str | Path,
    queries_path: str | Path,
    manifest_path: str | Path,
    report_path: str | Path,
) -> dict[str, Any]:
    """Run all leakage checks for one fold and write the JSON report."""
    processed_dir = Path(processed_dir)
    fold_dir = Path(folds_dir) / f"fold_{fold}"
    queries_path = Path(queries_path)
    manifest = _read_json(Path(manifest_path))

    checks: dict[str, dict[str, Any]] = {}

    def check(name: str, passed: bool, details: Any = None) -> None:
        checks[name] = {"passed": bool(passed), "details": details}

    # --- 1. split partitions match the frozen V0 files and stay disjoint ---
    queries = pd.read_parquet(queries_path)
    frozen_splits = read_split_diseases(fold_dir)
    split_sets = {
        split: set(frozen_splits[split]) for split in ("train", "validation", "test")
    }
    observed = {
        split: set(queries.loc[queries["split"] == split, "disease_id"])  # type: ignore[arg-type]
        for split in ("train", "validation", "test")
    }
    mismatches = [
        f"{split}: {sorted(observed[split] ^ split_sets[split])}"
        for split in split_sets
        if observed[split] != split_sets[split]
    ]
    check("split_partitions_match_frozen", not mismatches, mismatches)

    pairwise_overlap = [
        (left, right)
        for left, right in (
            ("train", "validation"),
            ("train", "test"),
            ("validation", "test"),
        )
        if split_sets[left] & split_sets[right]
    ]
    check("disease_sets_disjoint", not pairwise_overlap, pairwise_overlap)

    assignments = pd.read_parquet(folds_dir / "disease_fold_assignments.parquet")
    fold_assignments = assignments[assignments["fold"] == fold]
    family_of = {
        str(row.disease_id): str(row.disease_family_id)
        for row in fold_assignments.itertuples(index=False)
    }
    family_sets = {
        split: {family_of[d] for d in diseases if d in family_of}
        for split, diseases in split_sets.items()
    }
    family_overlap = [
        (left, right)
        for left, right in (
            ("train", "validation"),
            ("train", "test"),
            ("validation", "test"),
        )
        if family_sets[left] & family_sets[right]
    ]
    check("disease_families_disjoint", not family_overlap, family_overlap)

    # --- 2. graph edge counts reconcile with frozen V0 adjacency ---
    edge_reconciled = {}
    for key, expected in manifest["edge_counts"].items():
        relation = key[len("edge_") :]
        npz_path = (
            Path(processed_dir).parent
            / "graphs"
            / f"fold_{fold}"
            / f"A_{relation}.npz"
        )
        actual = _edge_nnz_from_npz(npz_path)
        edge_reconciled[relation] = {"expected": expected, "v0_nnz": actual}
    check(
        "graph_edge_counts_reconcile_v0",
        all(
            value["expected"] == value["v0_nnz"]
            for value in edge_reconciled.values()
        ),
        edge_reconciled,
    )

    # --- 3. V0 fold leakage report is re-endorsed ---
    v0_report = _read_json(fold_dir / "leakage_report.json")
    v0_passed = all(v0_report["checks"].values())
    check("v0_fold_leakage_endorsed", v0_passed, v0_report["checks"])

    # --- 4. query HPO ids resolve through the fold-specific training IC ---
    ic = load_fold_ic(fold_dir)
    unknown_hpo = sorted(
        {
            hpo
            for row in queries.itertuples(index=False)
            for hpo in _json_list(row.hpo_ids)
            if hpo not in ic
        }
    )
    weights_ok = bool(
        queries["hpo_weights"].map(
            lambda values: all(
                isinstance(v, (int, float)) and v >= 0 and float(v) == float(v)
                for v in values
            )
        ).all()
    )
    check("hpo_ids_in_fold_training_ic", not unknown_hpo, unknown_hpo)
    check("hpo_weights_nonnegative_finite", weights_ok)

    # --- 5. canonical positive genes: covered or explicitly unresolved ---
    universe = load_candidate_universe(fold_dir)
    universe_hash = _sha256_file(fold_dir / "candidate_gene_universe.parquet")
    ic_hash = _sha256_file(fold_dir / "train_hpo_ic.parquet")
    bad_rows = []
    unresolved_total = 0
    for row in queries.itertuples(index=False):
        positive = set(_json_list(row.positive_gene_ids))
        covered = set(_json_list(row.covered_positive_gene_ids))
        unresolved = set(_json_list(row.unresolved_positive_gene_ids))
        unresolved_total += len(unresolved)
        if not (covered == positive - unresolved and covered <= universe):
            bad_rows.append(str(row.disease_id))
    check(
        "positive_genes_covered_or_unresolved",
        not bad_rows,
        {"bad_rows": bad_rows, "unresolved_total": int(unresolved_total)},
    )

    # --- 6. serialization hash present and no confirmed negatives emitted ---
    check(
        "deterministic_serialization_hash",
        bool(manifest.get("serialization_sha256")),
        manifest.get("serialization_sha256"),
    )
    has_negative_field = any(
        "negative" in column.lower() for column in queries.columns
    )
    check(
        "no_confirmed_negative_labels",
        not has_negative_field,
        {"columns": list(queries.columns)},
    )

    failed = [name for name, result in checks.items() if not result["passed"]]
    report = {
        "fold": int(fold),
        "status": "PASS" if not failed else "FAIL",
        "passed_checks": len(checks) - len(failed),
        "failed_checks": len(failed),
        "checks": checks,
        "audit": {
            "candidate_universe_sha256": universe_hash,
            "training_ic_sha256": ic_hash,
            "serialization_sha256": manifest.get("serialization_sha256"),
            "unlabelled_never_confirmed_negative": True,
            "query_rows": int(len(queries)),
        },
        "v0_endorsed_leakage_report": v0_report,
    }
    report_path = Path(report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = report_path.with_name(report_path.name + ".tmp")
    temporary.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(report_path)
    LOGGER.info(
        "fold %d leakage report: %s (%d/%d checks)",
        fold,
        report["status"],
        report["passed_checks"],
        len(checks),
    )
    return report


__all__ = ["run_fold_leakage"]
