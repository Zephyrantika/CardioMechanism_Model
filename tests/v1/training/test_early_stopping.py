"""Tests for early stopping and checkpointing (Milestone 2)."""

from __future__ import annotations

import torch

from phenotype_network_v1.training.trainer import (
    EarlyStopping,
    describe_model,
    save_checkpoint,
)


def test_early_stopping_improving_never_stops() -> None:
    stopper = EarlyStopping(monitor="val_loss", mode="min", patience=3)
    for epoch in range(10):
        assert not stopper.update({"val_loss": 1.0 / (epoch + 1)}, epoch)
    assert stopper.best_epoch == 9


def test_early_stopping_patience_triggers() -> None:
    stopper = EarlyStopping(monitor="val_loss", mode="min", patience=2)
    assert not stopper.update({"val_loss": 1.0}, 0)
    assert not stopper.update({"val_loss": 1.5}, 1)  # worse, counter=1
    assert stopper.update({"val_loss": 1.6}, 2)  # counter=2 >= patience


def test_early_stopping_max_mode() -> None:
    stopper = EarlyStopping(monitor="acc", mode="max", patience=1)
    assert not stopper.update({"acc": 0.5}, 0)
    assert not stopper.update({"acc": 0.6}, 1)
    assert stopper.update({"acc": 0.55}, 2)  # not better -> stop


def test_early_stopping_rejects_bad_mode() -> None:
    try:
        EarlyStopping(monitor="x", mode="median")
    except ValueError:
        return
    raise AssertionError("expected ValueError for invalid mode")


def test_early_stopping_state_dict() -> None:
    stopper = EarlyStopping(monitor="val_loss", mode="min", patience=2)
    stopper.update({"val_loss": 1.0}, 0)
    state = stopper.state_dict()
    assert state["best"] == 1.0
    assert state["patience"] == 2


def test_describe_model_counts_parameters() -> None:
    model = torch.nn.Linear(4, 3)
    assert describe_model(model) == 4 * 3 + 3


def test_save_checkpoint_round_trip(tmp_path) -> None:  # noqa: ANN001
    model = torch.nn.Linear(4, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    path = save_checkpoint(
        path=tmp_path / "model.pt",
        model=model,
        optimizer=optimizer,
        epoch=3,
        seed=7,
        config_hash="cfg",
        data_hashes={"x": "y"},
        git_commit="abc",
        framework_versions={"torch": torch.__version__},
    )
    assert path.exists()
    manifest_path = path.with_suffix(".json")
    assert manifest_path.exists()
    import json

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["epoch"] == 3
    assert manifest["seed"] == 7
    assert manifest["config_hash"] == "cfg"
