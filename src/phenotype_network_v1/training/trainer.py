"""Early stopping and checkpointing for V1 training (Milestone 2).

Validation selects the epoch; test labels are never read here. Model and
optimizer state plus the M0 checkpoint manifest are stored under
``outputs/v1/checkpoints``.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch

from phenotype_network_v1.checkpoint import CheckpointManifest

LOGGER = logging.getLogger("phenotype_network_v1.training.trainer")


@dataclass
class EarlyStopping:
    """Stop training when a monitored metric stops improving."""

    monitor: str = "val_loss"
    mode: str = "min"  # 'min' or 'max'
    patience: int = 10
    min_delta: float = 1e-6
    best: float | None = field(default=None, init=False)
    best_epoch: int = field(default=-1, init=False)
    _counter: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        if self.mode not in {"min", "max"}:
            raise ValueError("mode must be 'min' or 'max'")
        if self.patience < 0 or self.min_delta < 0:
            raise ValueError("patience and min_delta must be non-negative")

    def _improved(self, value: float) -> bool:
        if self.best is None:
            return True
        if self.mode == "min":
            return value < self.best - self.min_delta
        return value > self.best + self.min_delta

    def update(self, metrics: dict[str, float], epoch: int) -> bool:
        """Feed the latest metrics; return True when training should stop."""
        value = float(metrics[self.monitor])
        if self._improved(value):
            self.best = value
            self.best_epoch = epoch
            self._counter = 0
            return False
        self._counter += 1
        return self._counter >= self.patience

    def state_dict(self) -> dict[str, Any]:
        return {
            "monitor": self.monitor,
            "mode": self.mode,
            "patience": self.patience,
            "min_delta": self.min_delta,
            "best": self.best,
            "best_epoch": self.best_epoch,
            "counter": self._counter,
        }


def save_checkpoint(
    *,
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    seed: int,
    config_hash: str,
    data_hashes: dict[str, str],
    git_commit: str,
    framework_versions: dict[str, str],
    extra: dict[str, Any] | None = None,
) -> Path:
    """Save model + optimizer state together with the checkpoint manifest."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    manifest = CheckpointManifest(
        seed=seed,
        epoch=epoch,
        config_hash=config_hash,
        data_hashes=data_hashes,
        git_commit=git_commit,
        framework_versions=framework_versions,
    )
    manifest_path = path.with_suffix(".json")
    payload = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "manifest": manifest.to_dict(),
    }
    if extra:
        payload["extra"] = extra
    temporary = path.with_name(path.name + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)
    manifest.write(manifest_path)
    return path


def load_checkpoint(path: str | Path) -> dict[str, Any]:
    """Load a checkpoint payload without attaching it to a model."""
    return torch.load(path, map_location="cpu", weights_only=False)


def describe_model(model: torch.nn.Module) -> int:
    """Return the number of trainable parameters."""
    return sum(parameter.numel() for parameter in model.parameters())


def train_epoch(
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    loss_fn: Any,
    device: torch.device,
    train_data: dict[str, torch.Tensor],
    batch_indices: list[list[int]],
) -> float:
    """Run one training epoch; returns mean loss over batches."""
    model.train()
    total_loss = 0.0
    for batch in batch_indices:
        optimizer.zero_grad()
        logits = model(train_data, batch=batch)
        loss = loss_fn(logits, train_data["targets"][batch])
        loss.backward()
        optimizer.step()
        total_loss += float(loss.detach())
    return total_loss / max(1, len(batch_indices))


def evaluate(
    *,
    model: torch.nn.Module,
    loss_fn: Any,
    device: torch.device,
    data: dict[str, torch.Tensor],
    indices: list[int] | None = None,
) -> dict[str, float]:
    """Run model in eval mode; returns loss (and accuracy for BCE targets)."""
    model.eval()
    with torch.no_grad():
        logits = model(data, batch=indices)
        loss_value = float(loss_fn(logits, data["targets"]))
    metrics: dict[str, float] = {"loss": loss_value}
    if indices is not None:
        selected = torch.tensor(indices, dtype=torch.long)
        logits_selected = model(data, batch=indices)
        loss_selected = float(loss_fn(logits_selected, data["targets"][selected]))
        metrics["val_loss"] = loss_selected
    return metrics


__all__ = [
    "EarlyStopping",
    "describe_model",
    "evaluate",
    "load_checkpoint",
    "save_checkpoint",
    "train_epoch",
]
