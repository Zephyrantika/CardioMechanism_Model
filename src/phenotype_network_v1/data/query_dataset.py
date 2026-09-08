"""Fold-specific query instances for V1 (Milestone 1).

Converts frozen V0 folds and disease annotations into auditable query
instances with the schema defined in V1_IMPLEMENTATION_PLAN.md section 6.3:

    fold, split, disease_id, disease_family_ids, hpo_ids, hpo_weights,
    positive_gene_ids, covered_positive_gene_ids,
    unresolved_positive_gene_ids, source_version

HPO weights are ``frequency_weight * fold-specific training IC``. Positive
genes use canonical NCBI Gene IDs. Unlabelled genes are never marked as
confirmed negatives anywhere in this module.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable

import pandas as pd

LOGGER = logging.getLogger("phenotype_network_v1.data.query_dataset")

QUERY_FIELDS = [
    "fold",
    "split",
    "disease_id",
    "disease_family_ids",
    "hpo_ids",
    "hpo_weights",
    "positive_gene_ids",
    "covered_positive_gene_ids",
    "unresolved_positive_gene_ids",
    "source_version",
]

SPLITS = ("train", "validation", "test")

_SOURCE_VERSION = "v0-folds-frozen"


def _read_disease_list(path: str | Path) -> list[str]:
    lines = [
        line.strip()
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return lines


def read_split_diseases(fold_dir: str | Path) -> dict[str, list[str]]:
    """Read train/validation/test disease IDs from the frozen fold directory.

    The V0 files are named ``train_diseases.txt``, ``val_diseases.txt`` and
    ``test_diseases.txt``; the returned mapping uses schema split names.
    """
    fold_dir = Path(fold_dir)
    name_to_split = {
        "train_diseases.txt": "train",
        "val_diseases.txt": "validation",
        "test_diseases.txt": "test",
    }
    splits: dict[str, list[str]] = {}
    for filename, split in name_to_split.items():
        path = fold_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"missing frozen disease split file: {path}")
        splits[split] = _read_disease_list(path)
    return splits


def load_disease_families(assignments_path: str | Path, fold: int) -> dict[str, list[str]]:
    """Map each disease to its MONDO family id for the given fold."""
    assignments = pd.read_parquet(assignments_path)
    rows = assignments[assignments["fold"] == fold]
    mapping: dict[str, list[str]] = {}
    for disease_id, family_id in rows[["disease_id", "disease_family_id"]].itertuples(
        index=False
    ):
        mapping[str(disease_id)] = [str(family_id)]
    return mapping


def load_fold_ic(fold_dir: str | Path) -> dict[str, float]:
    """Load the fold-specific training IC table."""
    path = Path(fold_dir) / "train_hpo_ic.parquet"
    if not path.exists():
        raise FileNotFoundError(f"missing fold IC table: {path}")
    table = pd.read_parquet(path)
    return {
        str(row.hpo_id): float(row.ic)
        for row in table.itertuples(index=False)
    }


def load_candidate_universe(fold_dir: str | Path) -> set[str]:
    path = Path(fold_dir) / "candidate_gene_universe.parquet"
    if not path.exists():
        raise FileNotFoundError(f"missing candidate universe: {path}")
    table = pd.read_parquet(path)
    return {str(gene_id) for gene_id in table["gene_id"].tolist()}


def _index_by_disease(
    table: pd.DataFrame, disease_column: str, value_columns: Iterable[str]
) -> dict[str, list[tuple[str, ...]]]:
    grouped: dict[str, list[tuple[str, ...]]] = {}
    for row in table.itertuples(index=False):
        values = tuple(
            str(getattr(row, column)) for column in value_columns
        )
        grouped.setdefault(str(getattr(row, disease_column)), []).append(values)
    return grouped


def build_fold_queries(
    fold: int,
    *,
    processed_dir: str | Path,
    folds_dir: str | Path,
    assignments_path: str | Path,
    source_version: str = _SOURCE_VERSION,
) -> pd.DataFrame:
    """Build query instances for one fold (train/validation/test splits).

    No query carries a confirmed-negative label. Genes outside the frozen
    candidate universe are listed under ``unresolved_positive_gene_ids`` and
    excluded from ``covered_positive_gene_ids``.
    """
    processed_dir = Path(processed_dir)
    fold_dir = Path(folds_dir) / f"fold_{fold}"

    phenotypes = pd.read_parquet(processed_dir / "cardiovascular_disease_phenotypes.parquet")
    genes = pd.read_parquet(processed_dir / "cardiovascular_disease_genes.parquet")

    phenotype_rows = _index_by_disease(
        phenotypes, "disease_id", ("hpo_id", "frequency_weight")
    )
    gene_rows = _index_by_disease(genes, "disease_id", ("gene_id",))

    splits = read_split_diseases(fold_dir)
    families = load_disease_families(assignments_path, fold)
    ic = load_fold_ic(fold_dir)
    universe = load_candidate_universe(fold_dir)

    records: list[dict[str, object]] = []
    for split, diseases in splits.items():
        for disease_id in diseases:
            family_ids = families.get(disease_id, [])
            hpo_rows = phenotype_rows.get(disease_id, [])
            hpo_ids: list[str] = []
            hpo_weights: list[float] = []
            for hpo_id, frequency_weight in hpo_rows:
                if hpo_id not in ic:
                    raise KeyError(
                        f"fold {fold} {disease_id}: HPO {hpo_id} missing from "
                        f"fold-specific training IC table"
                    )
                hpo_ids.append(hpo_id)
                hpo_weights.append(float(frequency_weight) * ic[hpo_id])

            positive_ids = [gene_id for (gene_id,) in gene_rows.get(disease_id, [])]
            covered = [gene_id for gene_id in positive_ids if gene_id in universe]
            unresolved = [gene_id for gene_id in positive_ids if gene_id not in universe]

            records.append(
                {
                    "fold": int(fold),
                    "split": split,
                    "disease_id": disease_id,
                    "disease_family_ids": family_ids,
                    "hpo_ids": hpo_ids,
                    "hpo_weights": hpo_weights,
                    "positive_gene_ids": positive_ids,
                    "covered_positive_gene_ids": covered,
                    "unresolved_positive_gene_ids": unresolved,
                    "source_version": source_version,
                }
            )

    frame = pd.DataFrame(records, columns=QUERY_FIELDS)
    LOGGER.info(
        "fold %d query instances: train=%d validation=%d test=%d",
        fold,
        len(splits["train"]),
        len(splits["validation"]),
        len(splits["test"]),
    )
    return frame


def write_query_instances(frame: pd.DataFrame, destination: str | Path) -> Path:
    """Atomically write query instances as Parquet."""
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    frame.to_parquet(temporary, index=False)
    temporary.replace(destination)
    return destination


__all__ = [
    "QUERY_FIELDS",
    "SPLITS",
    "build_fold_queries",
    "load_candidate_universe",
    "load_disease_families",
    "load_fold_ic",
    "read_split_diseases",
    "write_query_instances",
]
