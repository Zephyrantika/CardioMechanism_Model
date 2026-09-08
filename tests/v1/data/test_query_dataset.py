"""Tests for V1 query-instance construction (Milestone 1).

Uses small synthetic fixtures only; no real V0 data is required.
"""

from __future__ import annotations

import pandas as pd
import pytest

from phenotype_network_v1.data.query_dataset import (
    QUERY_FIELDS,
    build_fold_queries,
    read_split_diseases,
)


def _write_parquet(path, frame: pd.DataFrame) -> None:  # noqa: ANN001
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)


def make_fold_layout(tmp_path) -> dict:  # noqa: ANN001
    """Synthetic fold layout: d1 train, d2 test, families A/B disjoint."""
    processed = tmp_path / "processed"
    folds = tmp_path / "folds"
    fold_dir = folds / "fold_0"
    fold_dir.mkdir(parents=True, exist_ok=True)

    _write_parquet(
        processed / "cardiovascular_disease_phenotypes.parquet",
        pd.DataFrame(
            {
                "disease_id": ["d1", "d1", "d2"],
                "hpo_id": ["HP:1", "HP:2", "HP:3"],
                "frequency_weight": [0.5, 1.0, 0.2],
            }
        ),
    )
    _write_parquet(
        processed / "cardiovascular_disease_genes.parquet",
        pd.DataFrame(
            {
                "disease_id": ["d1", "d1", "d2"],
                "gene_id": ["100", "200", "300"],
            }
        ),
    )
    _write_parquet(
        folds / "disease_fold_assignments.parquet",
        pd.DataFrame(
            {
                "disease_id": ["d1", "d2"],
                "disease_family_id": ["FAM-A", "FAM-B"],
                "fold": [0, 0],
            }
        ),
    )
    (fold_dir / "train_diseases.txt").write_text("d1\n", encoding="utf-8")
    (fold_dir / "val_diseases.txt").write_text("", encoding="utf-8")
    (fold_dir / "test_diseases.txt").write_text("d2\n", encoding="utf-8")
    _write_parquet(
        fold_dir / "train_hpo_ic.parquet",
        pd.DataFrame(
            {
                "hpo_id": ["HP:1", "HP:2", "HP:3"],
                "ic": [1.0, 2.0, 0.5],
            }
        ),
    )
    _write_parquet(
        fold_dir / "candidate_gene_universe.parquet",
        pd.DataFrame({"gene_id": ["100", "300"]}),
    )
    return {
        "root": tmp_path,
        "processed": processed,
        "folds": folds,
        "fold_dir": fold_dir,
    }


def test_split_disease_reading(tmp_path) -> None:  # noqa: ANN001
    layout = make_fold_layout(tmp_path)
    splits = read_split_diseases(layout["fold_dir"])
    assert splits["train"] == ["d1"]
    assert splits["validation"] == []
    assert splits["test"] == ["d2"]


def test_build_fold_queries_schema_and_weights(tmp_path) -> None:  # noqa: ANN001
    layout = make_fold_layout(tmp_path)
    frame = build_fold_queries(
        0,
        processed_dir=layout["processed"],
        folds_dir=layout["folds"],
        assignments_path=layout["folds"] / "disease_fold_assignments.parquet",
    )
    assert list(frame.columns) == QUERY_FIELDS
    assert len(frame) == 2  # one train + one test query

    train = frame[frame["split"] == "train"].iloc[0]
    assert train["disease_id"] == "d1"
    assert train["hpo_ids"] == ["HP:1", "HP:2"]
    assert train["hpo_weights"] == [0.5 * 1.0, 1.0 * 2.0]
    assert train["disease_family_ids"] == ["FAM-A"]
    assert train["covered_positive_gene_ids"] == ["100"]
    assert train["unresolved_positive_gene_ids"] == ["200"]

    test_row = frame[frame["split"] == "test"].iloc[0]
    assert test_row["disease_id"] == "d2"
    assert test_row["covered_positive_gene_ids"] == ["300"]
    assert test_row["unresolved_positive_gene_ids"] == []
    assert test_row["source_version"] == "v0-folds-frozen"


def test_missing_ic_raises(tmp_path) -> None:  # noqa: ANN001
    layout = make_fold_layout(tmp_path)
    # Remove HP:2 from the IC table so d1's second phenotype cannot resolve.
    ic_path = layout["fold_dir"] / "train_hpo_ic.parquet"
    reduced = pd.read_parquet(ic_path)
    reduced = reduced[reduced["hpo_id"] != "HP:2"]
    reduced.to_parquet(ic_path, index=False)
    with pytest.raises(KeyError):
        build_fold_queries(
            0,
            processed_dir=layout["processed"],
            folds_dir=layout["folds"],
            assignments_path=layout["folds"] / "disease_fold_assignments.parquet",
        )


def test_positive_never_negative(tmp_path) -> None:  # noqa: ANN001
    layout = make_fold_layout(tmp_path)
    frame = build_fold_queries(
        0,
        processed_dir=layout["processed"],
        folds_dir=layout["folds"],
        assignments_path=layout["folds"] / "disease_fold_assignments.parquet",
    )
    columns = [column.lower() for column in frame.columns]
    assert not any("negative" in column for column in columns)
