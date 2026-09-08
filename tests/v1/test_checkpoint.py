"""Tests for the V1 checkpoint manifest foundations (Milestone 0).

Pure-standard-library tests: no PyTorch dependency, synthetic fixtures only.
"""

from __future__ import annotations

import json

from phenotype_network_v1.checkpoint import (
    SCHEMA_VERSION,
    CheckpointManifest,
    git_commit,
    hash_mapping,
    sha256_bytes,
    sha256_file,
)


def test_sha256_bytes_known_vector() -> None:
    # SHA-256 of the empty string.
    assert sha256_bytes(b"") == (
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    )


def test_sha256_bytes_hello() -> None:
    assert sha256_bytes(b"hello") == (
        "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824"
    )


def test_sha256_file_matches_bytes(tmp_path) -> None:  # noqa: ANN001
    path = tmp_path / "blob.bin"
    payload = b"phenotype-network-v1\n" * 1000
    path.write_bytes(payload)
    assert sha256_file(path) == sha256_bytes(payload)


def test_hash_mapping_stable() -> None:
    first = hash_mapping({"b": 2, "a": [1, 2]})
    second = hash_mapping({"a": [1, 2], "b": 2})
    assert first == second
    assert first != hash_mapping({"a": [1, 2], "b": 3})


def test_hash_mapping_empty_and_none() -> None:
    empty = hash_mapping({})
    assert empty == hash_mapping(None)
    assert empty == sha256_bytes(b"{}")


def test_manifest_write_read_round_trip(tmp_path) -> None:  # noqa: ANN001
    destination = tmp_path / "checkpoint.json"
    manifest = CheckpointManifest(
        seed=42,
        epoch=7,
        config_hash=hash_mapping({"learning_rate": 0.001}),
        data_hashes={"folds": sha256_bytes(b"frozen")},
        git_commit="abc123",
        framework_versions={"python": "3.11"},
    )
    written = manifest.write(destination)
    assert written == destination
    assert destination.exists()
    loaded = CheckpointManifest.read(destination)
    assert loaded.schema_version == SCHEMA_VERSION
    assert loaded.seed == 42
    assert loaded.epoch == 7
    assert loaded.config_hash == manifest.config_hash
    assert loaded.data_hashes == {"folds": sha256_bytes(b"frozen")}
    assert loaded.git_commit == "abc123"
    assert loaded.framework_versions == {"python": "3.11"}


def test_manifest_write_is_atomic_no_tmp_left(tmp_path) -> None:  # noqa: ANN001
    destination = tmp_path / "checkpoint.json"
    CheckpointManifest(seed=1).write(destination)
    leftovers = list(tmp_path.glob("*.tmp"))
    assert leftovers == []
    raw = json.loads(destination.read_text(encoding="utf-8"))
    assert raw["schema_version"] == SCHEMA_VERSION
    assert "created_at" in raw


def test_git_commit_returns_unknown_outside_repo(tmp_path) -> None:  # noqa: ANN001
    assert git_commit(tmp_path) == "unknown"


def test_git_commit_returns_hash_in_repo() -> None:
    # The V1 package tree lives inside a Git repository (main/codex/v1).
    root = __import__("pathlib").Path(__file__).resolve().parents[2]
    commit = git_commit(root)
    assert len(commit) == 40
    assert all(char in "0123456789abcdef" for char in commit)
