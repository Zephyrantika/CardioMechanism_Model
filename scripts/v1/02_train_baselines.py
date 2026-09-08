"""Train V1 deep baselines (Milestone 2).

Trains one baseline model (gcn | rgcn | hgt) on one fold using the M1 query
instances and PyG graph artifacts. Validation selects the epoch; test labels
are never read during training. Checkpoints carry the M0 manifest.

Usage (server, authoritative):
    python scripts/v1/02_train_baselines.py --model rgcn --fold 0 --device cuda

CPU smoke (development machine):
    python scripts/v1/02_train_baselines.py --model gcn --fold 0 --smoke
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import torch

from phenotype_network_v1.checkpoint import git_commit, hash_mapping
from phenotype_network_v1.models.gcn import GCNBaseline, build_unified_from_stores
from phenotype_network_v1.models.hgt import HGTBaseline
from phenotype_network_v1.models.rgcn import RGCNBaseline
from phenotype_network_v1.training.sampling import seed_everything
from phenotype_network_v1.training.trainer import (
    EarlyStopping,
    describe_model,
    save_checkpoint,
)

LOGGER = logging.getLogger("v1.train_baselines")

MODEL_FACTORIES: dict[str, Any] = {
    "gcn": GCNBaseline,
    "rgcn": RGCNBaseline,
    "hgt": HGTBaseline,
}
NODE_TYPE_ORDER = ("phenotype", "gene", "pathway")


def _configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )


def load_fold_inputs(root: Path, fold: int) -> dict[str, Any]:
    heterodata_path = root / "data" / "graphs" / "v1" / f"fold_{fold}" / "heterodata.pt"
    queries_path = root / "data" / "processed" / "v1" / f"fold_{fold}" / "query_instances.parquet"
    node_map_path = root / "data" / "graphs" / f"fold_{fold}" / "node_map.parquet"
    heterodata = torch.load(heterodata_path, weights_only=False)
    queries = pd.read_parquet(queries_path)
    node_map = pd.read_parquet(node_map_path).sort_values("node_index")

    # Robust mapping: enumerate phenotype/gene rows in node order.
    phenotype_rows = node_map[node_map["node_type"] == "phenotype"].reset_index(drop=True)
    hpo_to_local = {str(row.node_id): int(position) for position, row in phenotype_rows.iterrows()}
    gene_rows = node_map[node_map["node_type"] == "gene"].reset_index(drop=True)
    gene_id_to_local = {
        str(row.node_id): int(position) for position, row in gene_rows.iterrows()
    }
    n_gene = len(gene_rows)
    graph = build_unified_from_stores(heterodata)
    gene_node_indices = list(
        range(int(graph["offsets"]["gene"]), int(graph["offsets"]["gene"]) + n_gene)
    )
    return {
        "graph": graph,
        "queries": queries,
        "hpo_to_local": hpo_to_local,
        "gene_id_to_local": gene_id_to_local,
        "n_gene": n_gene,
        "gene_node_indices": gene_node_indices,
        "hpo_offset": int(graph["offsets"]["phenotype"]),
        "data_hashes": {
            "query_instances": hash_mapping(
                {"fold": int(fold), "file": queries_path.name}
            ),
            "heterodata": heterodata_path.name,
        },
    }


def build_targets(
    split_frame: pd.DataFrame, gene_id_to_local: dict[str, int], n_gene: int
) -> torch.Tensor:
    targets = torch.zeros(len(split_frame), n_gene)
    for position, row in enumerate(split_frame.itertuples(index=False)):
        for gene_id in row.covered_positive_gene_ids:
            local = gene_id_to_local.get(str(gene_id))
            if local is not None:
                targets[position, local] = 1.0
    return targets


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=tuple(MODEL_FACTORIES), required=True)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu", help="cpu or cuda")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="CPU smoke run: 2 epochs over a small query subset.",
    )
    parser.add_argument(
        "--limit-queries", type=int, default=None, help="Cap training queries."
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)
    _configure_logging(args.log_level)

    root = Path(__file__).resolve().parents[2]
    seed_everything(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")

    fold_inputs = load_fold_inputs(root, args.fold)
    graph = fold_inputs["graph"]
    queries = fold_inputs["queries"]
    gene_id_to_local = fold_inputs["gene_id_to_local"]
    n_gene = fold_inputs["n_gene"]

    train_frame = queries[queries["split"] == "train"]
    val_frame = queries[queries["split"] == "validation"]
    if args.limit_queries:
        train_frame = train_frame.head(args.limit_queries)
    if args.smoke:
        train_frame = train_frame.head(5)
        val_frame = val_frame.head(5)
        args.epochs = min(args.epochs, 2)

    train_targets = build_targets(train_frame, gene_id_to_local, n_gene)
    val_targets = build_targets(val_frame, gene_id_to_local, n_gene)
    loss_fn = torch.nn.BCEWithLogitsLoss()

    factory_kwargs: dict[str, Any] = {
        "num_nodes": int(graph["num_nodes"]),
        "gene_node_indices": fold_inputs["gene_node_indices"],
        "hidden_dim": args.hidden_dim,
        "num_layers": args.num_layers,
        "seed": args.seed,
    }
    if args.model == "rgcn":
        factory_kwargs["num_relations"] = int(graph["num_relations"])
    if args.model == "hgt":
        factory_kwargs["num_relations"] = int(graph["num_relations"])
    model = MODEL_FACTORIES[args.model](**factory_kwargs).to(device)
    LOGGER.info(
        "model=%s fold=%d params=%d device=%s queries=%d genes=%d",
        args.model,
        args.fold,
        describe_model(model),
        device,
        len(train_frame),
        n_gene,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    early_stopping = EarlyStopping(patience=10 if not args.smoke else 1)

    history: list[dict[str, float]] = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        for position in range(len(train_frame)):
            optimizer.zero_grad()
            logits = model(
                graph,
                _hpo_seed(train_frame, position, fold_inputs["hpo_to_local"]),
                fold_inputs["hpo_offset"],
            )
            loss = loss_fn(logits, train_targets[position])
            loss.backward()
            optimizer.step()
            total += float(loss.detach())
        model.eval()
        with torch.no_grad():
            val_loss = 0.0
            for position in range(min(len(val_frame), 20)):
                logits = model(
                    graph,
                    _hpo_seed(val_frame, position, fold_inputs["hpo_to_local"]),
                    fold_inputs["hpo_offset"],
                )
                val_loss += float(loss_fn(logits, val_targets[position]))
        val_loss /= max(1, min(len(val_frame), 20))
        history.append({"epoch": epoch, "train_loss": total / len(train_frame), "val_loss": val_loss})
        stop = early_stopping.update({"val_loss": val_loss}, epoch)
        LOGGER.info(
            "epoch=%d train_loss=%.4f val_loss=%.4f best=%s",
            epoch,
            history[-1]["train_loss"],
            val_loss,
            early_stopping.best_epoch,
        )
        if stop:
            LOGGER.info("early stopping at epoch %d", epoch)
            break

    output_dir = args.output_dir or (root / "outputs" / "v1" / "baselines" / args.model / f"fold_{args.fold}")
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "model.pt"
    commit = git_commit(root)
    save_checkpoint(
        path=checkpoint_path,
        model=model,
        optimizer=optimizer,
        epoch=early_stopping.best_epoch,
        seed=args.seed,
        config_hash=hash_mapping(vars(args)),
        data_hashes=fold_inputs["data_hashes"],
        git_commit=commit,
        framework_versions={"torch": torch.__version__},
        extra={"model": args.model, "fold": args.fold, "history": history},
    )
    summary = {
        "model": args.model,
        "fold": args.fold,
        "seed": args.seed,
        "params": describe_model(model),
        "best_epoch": early_stopping.best_epoch,
        "best_val_loss": early_stopping.best,
        "history": history,
        "checkpoint": str(checkpoint_path),
        "git_commit": commit,
        "device": str(device),
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    LOGGER.info("checkpoint and summary written under %s", output_dir)
    return 0


def _hpo_seed(frame: pd.DataFrame, position: int, hpo_to_local: dict[str, int]) -> list[int]:
    row = frame.iloc[position]
    return [hpo_to_local[str(hpo)] for hpo in row.hpo_ids if str(hpo) in hpo_to_local]


if __name__ == "__main__":
    sys.exit(main())
