"""Checkpoint manifest foundations for V1 (Milestone 0).

Defines the reproducible-run metadata schema and atomic JSON storage. Model
and optimizer state serialisation (PyTorch ``torch.save``) is added by later
milestones; this module only covers the parts every checkpoint must carry:

* schema version, created time, seed, epoch;
* configuration hash and data hashes (SHA-256);
* Git commit of the running tree;
* framework/CUDA versions recorded when available.

No dependency on PyTorch is required at import time.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

SCHEMA_VERSION = 1


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    """Compute the SHA-256 of a file without loading it into memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while True:
            chunk = stream.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def hash_mapping(mapping: Mapping[str, Any] | None) -> str:
    """Stable SHA-256 of a JSON-serialisable mapping (config or data hashes)."""
    if not mapping:
        return sha256_bytes(b"{}")
    canonical = json.dumps(
        mapping, sort_keys=True, separators=(",", ":"), default=str
    )
    return sha256_bytes(canonical.encode("utf-8"))


def git_commit(repo_root: str | Path) -> str:
    """Return the current Git commit hash of ``repo_root``.

    Falls back to ``"unknown"`` when the directory is not a Git repository,
    Git is unavailable, or the call fails.
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo_root),
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    if result.returncode != 0:
        return "unknown"
    return result.stdout.strip()


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass
class CheckpointManifest:
    """Metadata every V1 checkpoint must record."""

    schema_version: int = SCHEMA_VERSION
    created_at: str = field(default_factory=_utc_now_iso)
    seed: int | None = None
    epoch: int | None = None
    config_hash: str | None = None
    data_hashes: dict[str, str] = field(default_factory=dict)
    git_commit: str | None = None
    framework_versions: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def write(self, path: str | Path) -> Path:
        """Atomically write the manifest as pretty JSON (tmp + replace)."""
        destination = Path(path)
        temporary = destination.with_name(destination.name + ".tmp")
        temporary.write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(destination)
        return destination

    @classmethod
    def read(cls, path: str | Path) -> "CheckpointManifest":
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(**{key: raw[key] for key in cls.__dataclass_fields__ if key in raw})  # type: ignore[return-value]


__all__ = [
    "CheckpointManifest",
    "SCHEMA_VERSION",
    "git_commit",
    "hash_mapping",
    "sha256_bytes",
    "sha256_file",
]
