"""Fold-specific PyG graph artifacts for V1 (Milestone 1).

Converts a frozen V0 graph directory (``node_map.parquet`` plus the
``A_<src>_<dst>.npz`` CSR adjacency blocks) into a PyG ``HeteroData`` with
node types ``phenotype``, ``gene`` and ``pathway``. Edge weights and the V0
global node indices are preserved so later milestones can reconcile against
the frozen V0 transition matrices and evaluate on the identical candidate
universe. Serialization is deterministic and hashed for reproducibility.
"""

from __future__ import annotations

import io
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch

LOGGER = logging.getLogger("phenotype_network_v1.data.pyg_graph")

NODE_TYPES = ("phenotype", "gene", "pathway")
RELATION_PREFIX = "A_"
# V0 graph files name phenotype nodes "hpo"; V1 node type is "phenotype".
_TYPE_ALIASES = {"hpo": "phenotype"}


def _canonical_node_type(name: str) -> str:
    return _TYPE_ALIASES.get(name, name)


def load_adjacency(path: str | Path) -> sp.csr_matrix:
    """Load a scipy CSR matrix saved by V0 (indices/indptr/data/format)."""
    archive = np.load(path, allow_pickle=False)
    shape = tuple(int(x) for x in archive["shape"])
    matrix = sp.csr_matrix(
        (archive["data"], archive["indices"], archive["indptr"]), shape=shape
    )
    return matrix


def _node_type_counts(node_map: pd.DataFrame) -> dict[str, int]:
    counts = node_map["node_type"].value_counts().to_dict()
    return {str(key): int(value) for key, value in counts.items()}


def build_heterodata(graph_dir: str | Path):
    """Build a PyG HeteroData object from a frozen V0 graph directory."""
    try:
        from torch_geometric.data import HeteroData
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "torch_geometric is required to build V1 graph artifacts; "
            "install the 'deep' dependency group"
        ) from exc

    graph_dir = Path(graph_dir)
    node_map = pd.read_parquet(graph_dir / "node_map.parquet")
    node_map = node_map.sort_values("node_index").reset_index(drop=True)
    assert node_map["node_index"].tolist() == list(range(len(node_map)))

    data = HeteroData()
    node_type_of_global = {  # global node index -> node type
        int(row.node_index): str(row.node_type)
        for row in node_map.itertuples(index=False)
    }
    for node_type in NODE_TYPES:
        indices = node_map.index[node_map["node_type"] == node_type].tolist()
        data[node_type].num_nodes = len(indices)
        # Global V0 node indices for this type, in local order.
        data[node_type].v0_node_index = torch.tensor(
            [int(node_map.loc[i, "node_index"]) for i in indices], dtype=torch.long
        )

    for path in sorted(graph_dir.glob(f"{RELATION_PREFIX}*.npz")):
        relation = path.name[len(RELATION_PREFIX) : -len(".npz")]
        src_type, dst_type = (
            _canonical_node_type(name)
            for name in relation.split("_", 1)
        )
        if src_type not in NODE_TYPES or dst_type not in NODE_TYPES:
            raise ValueError(f"unexpected relation file {path.name}")
        matrix = load_adjacency(path)
        coo = matrix.tocoo()
        if coo.nnz == 0:
            continue
        # Relate the global CSR coordinates to per-type local indices.
        local_index = {node_type: None for node_type in NODE_TYPES}
        for node_type in NODE_TYPES:
            mask = node_map["node_type"] == node_type
            index_of_global = {
                int(node_map.loc[i, "node_index"]): local
                for local, i in enumerate(node_map.index[mask])
            }
            local_index[node_type] = index_of_global
        edge_index = torch.tensor(
            [
                [
                    local_index[src_type].get(int(i), -1)
                    for i in coo.row.tolist()
                ],
                [
                    local_index[dst_type].get(int(j), -1)
                    for j in coo.col.tolist()
                ],
            ],
            dtype=torch.long,
        )
        data[src_type, relation, dst_type].edge_index = edge_index
        data[src_type, relation, dst_type].edge_weight = torch.tensor(
            coo.data.tolist(), dtype=torch.float32
        )

    data.fold = None  # assigned by caller through metadata
    return data


def heterodata_bytes(data: Any) -> bytes:
    """Serialize HeteroData deterministically to bytes (torch.save)."""
    buffer = io.BytesIO()
    torch.save(data, buffer)
    return buffer.getvalue()


def sha256_heterodata(data: Any) -> str:
    import hashlib

    return hashlib.sha256(heterodata_bytes(data)).hexdigest()


def _sha256_file(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_graph_manifest(
    fold: int,
    graph_dir: str | Path,
    data: Any,
    *,
    git_commit: str | None,
    source_inputs: dict[str, str],
) -> dict[str, Any]:
    """Aggregate node/edge counts and hashes into the graph manifest."""
    graph_dir = Path(graph_dir)
    counts: dict[str, int] = {}
    for node_type in NODE_TYPES:
        counts[f"node_{node_type}"] = int(data[node_type].num_nodes)
    for key in data.edge_types:
        relation = key[1]
        counts[f"edge_{relation}"] = int(data[key].edge_index.size(1))
    manifest: dict[str, Any] = {
        "fold": int(fold),
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "node_counts": {
            node_type: counts[f"node_{node_type}"] for node_type in NODE_TYPES
        },
        "edge_counts": {
            key: value
            for key, value in counts.items()
            if key.startswith("edge_")
        },
        "total_nodes": int(sum(counts[f"node_{t}"] for t in NODE_TYPES)),
        "serialization_sha256": sha256_heterodata(data),
        "git_commit": git_commit,
        "source_inputs": source_inputs,
    }
    return manifest


def write_graph_artifacts(
    fold: int,
    graph_dir: str | Path,
    destination_dir: str | Path,
    *,
    git_commit: str | None,
) -> dict[str, Any]:
    """Write heterodata.pt, graph_manifest.json; return the manifest."""
    graph_dir = Path(graph_dir)
    destination_dir = Path(destination_dir)
    destination_dir.mkdir(parents=True, exist_ok=True)

    source_inputs = {
        "node_map.parquet": _sha256_file(graph_dir / "node_map.parquet"),
        "graph_qc.json": _sha256_file(graph_dir / "graph_qc.json"),
    }
    for path in sorted(graph_dir.glob(f"{RELATION_PREFIX}*.npz")):
        source_inputs[path.name] = _sha256_file(path)

    data = build_heterodata(graph_dir)
    data.fold = fold

    bytes_path = destination_dir / "heterodata.pt"
    torch.save(data, bytes_path)
    LOGGER.info(
        "fold %d heterodata saved: nodes=%d edges=%d",
        fold,
        int(sum(data[node_type].num_nodes for node_type in NODE_TYPES)),
        sum(int(data[key].edge_index.size(1)) for key in data.edge_types),
    )

    manifest = build_graph_manifest(
        fold, graph_dir, data, git_commit=git_commit, source_inputs=source_inputs
    )
    manifest_path = destination_dir / "graph_manifest.json"
    temporary = manifest_path.with_name(manifest_path.name + ".tmp")
    temporary.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(manifest_path)
    return manifest


__all__ = [
    "NODE_TYPES",
    "RELATION_PREFIX",
    "build_graph_manifest",
    "build_heterodata",
    "heterodata_bytes",
    "load_adjacency",
    "sha256_heterodata",
    "write_graph_artifacts",
]
