"""Train the V1 phenotype-conditioned primary model (Milestone 3).

Composes QueryPhenotypeEncoder + ConditionedRGCN + ScoreDecoder with the
frozen V0 RWR residual. Ablations required by the plan are selectable via
``--ablation``:

    full | no_query_conditioning | condition_decoder_only |
    no_recurrent_seed_injection | weighted_mean_hpo_pool | no_rwr_residual

Validation selects epochs; test labels are never read during training.
Usage (server):  python scripts/v1/03_train_conditioned_model.py --fold 0 --device cuda
CPU smoke:       python scripts/v1/03_train_conditioned_model.py --fold 0 --smoke
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
from phenotype_network_v1.models.conditioned_rgcn import ConditionedRGCN
from phenotype_network_v1.models.decoder import ScoreDecoder
from phenotype_network_v1.models.gcn import build_unified_from_stores
from phenotype_network_v1.models.losses import classification_loss
from phenotype_network_v1.models.phenotype_encoder import QueryPhenotypeEncoder
from phenotype_network_v1.training.pu import sample_unlabelled, targets_from_sample
from phenotype_network_v1.training.sampling import derive_seed, seed_everything
from phenotype_network_v1.training.trainer import (
    EarlyStopping,
    describe_model,
    save_checkpoint,
)

LOGGER = logging.getLogger("v1.train_conditioned")

ABLATIONS = (
    "full",
    "no_query_conditioning",
    "condition_decoder_only",
    "no_recurrent_seed_injection",
    "weighted_mean_hpo_pool",
    "no_rwr_residual",
)


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
    phenotype_rows = node_map[node_map["node_type"] == "phenotype"].reset_index(drop=True)
    gene_rows = node_map[node_map["node_type"] == "gene"].reset_index(drop=True)
    hpo_to_local = {str(row.node_id): int(pos) for pos, row in phenotype_rows.iterrows()}
    gene_id_to_local = {str(row.node_id): int(pos) for pos, row in gene_rows.iterrows()}
    graph = build_unified_from_stores(heterodata)
    n_gene = len(gene_rows)
    gene_node_indices = list(
        range(int(graph["offsets"]["gene"]), int(graph["offsets"]["gene"]) + n_gene)
    )
    # Gene degrees from the frozen graph (in-degree of gene nodes).
    edge_index = graph["edge_index"]
    degrees = torch.zeros(int(graph["num_nodes"]), dtype=torch.long)
    degrees.index_add_(0, edge_index[1], torch.ones(edge_index.size(1), dtype=torch.long))
    gene_degrees = [int(degrees[g]) for g in gene_node_indices]
    # Frozen V0 RWR residual (log1p). Missing disease/gene records map to 0.
    rwr_path = root / "outputs" / "rankings" / f"fold_{fold}" / "rwr" / "rankings.parquet"
    rwr_by_disease: dict[str, dict[str, float]] = {}
    if rwr_path.exists():
        rwr = pd.read_parquet(rwr_path)
        for disease_id, gene_id, score in rwr[
            ["disease_id", "gene_id", "raw_rwr_score"]
        ].itertuples(index=False):
            rwr_by_disease.setdefault(str(disease_id), {})[str(gene_id)] = float(score)
    return {
        "graph": graph,
        "queries": queries,
        "hpo_to_local": hpo_to_local,
        "gene_id_to_local": gene_id_to_local,
        "n_gene": n_gene,
        "gene_node_indices": gene_node_indices,
        "hpo_offset": int(graph["offsets"]["phenotype"]),
        "rwr_by_disease": rwr_by_disease,
        "rwr_missing_count": 0,
        "gene_degrees": gene_degrees,
    }


def _query_inputs(
    frame: pd.DataFrame,
    position: int,
    hpo_to_local: dict[str, int],
    gene_id_to_local: dict[str, int],
    n_gene: int,
    rwr_by_disease: dict[str, dict[str, float]],
) -> dict[str, torch.Tensor]:
    row = frame.iloc[position]
    hpo_indices = [
        hpo_to_local[str(hpo)] for hpo in row.hpo_ids if str(hpo) in hpo_to_local
    ]
    weights = []
    for hpo, weight in zip(row.hpo_ids, row.hpo_weights):
        if str(hpo) in hpo_to_local:
            weights.append(float(weight))
    targets = torch.zeros(n_gene)
    for gene_id in row.covered_positive_gene_ids:
        local = gene_id_to_local.get(str(gene_id))
        if local is not None:
            targets[local] = 1.0
    rwr_feature = torch.zeros(n_gene)
    rwr = rwr_by_disease.get(str(row.disease_id), {})
    for gene_id, score in rwr.items():
        local = gene_id_to_local.get(str(gene_id))
        if local is not None:
            rwr_feature[local] = float(__import__("numpy").log1p(score))
    return {
        "hpo_indices": torch.tensor(hpo_indices, dtype=torch.long),
        "hpo_weights": torch.tensor(weights, dtype=torch.float32),
        "seed_hpo": hpo_indices,
        "targets": targets,
        "rwr_feature": rwr_feature,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--ablation", choices=ABLATIONS, default="full")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--relation-bases", type=int, default=8)
    parser.add_argument("--rho", type=float, default=0.20)
    parser.add_argument("--dropout", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--pu",
        action="store_true",
        help="Use degree-stratified sampled-unlabelled records as negatives.",
    )
    parser.add_argument("--ratio", type=int, default=20, help="PU sample ratio.")
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument(
        "--loss-focus",
        choices=("all", "positive_only"),
        default="all",
        help="all = BCE over positives+sampled; positive_only = weight 1 only for positives.",
    )
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)
    _configure_logging(args.log_level)

    root = Path(__file__).resolve().parents[2]
    seed_everything(args.seed)
    device = torch.device(
        args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"
    )
    inputs = load_fold_inputs(root, args.fold)
    graph = inputs["graph"]
    queries = inputs["queries"]
    train_frame = queries[queries["split"] == "train"]
    val_frame = queries[queries["split"] == "validation"]
    if args.smoke:
        train_frame = train_frame.head(4)
        val_frame = val_frame.head(4)
        args.epochs = min(args.epochs, 2)

    encoder = QueryPhenotypeEncoder(
        int(graph["sizes"]["phenotype"]),
        args.hidden_dim,
        attention=(args.ablation != "weighted_mean_hpo_pool"),
        seed=args.seed,
    )
    backbone = ConditionedRGCN(
        int(graph["num_nodes"]),
        int(graph["num_relations"]),
        hidden_dim=args.hidden_dim,
        num_layers=2,
        relation_bases=args.relation_bases,
        rho=args.rho,
        dropout=args.dropout,
        seed=args.seed,
    )
    decoder = ScoreDecoder(
        args.hidden_dim,
        use_rwr=(args.ablation != "no_rwr_residual"),
        seed=args.seed,
    )
    model = torch.nn.ModuleList([encoder, backbone, decoder]).to(device)
    total_params = sum(describe_model(module) for module in model)
    condition = args.ablation not in {"no_query_conditioning", "condition_decoder_only"}
    seed_injection = args.ablation not in {
        "no_query_conditioning",
        "condition_decoder_only",
        "no_recurrent_seed_injection",
    }
    decoder_query_on = args.ablation not in {"no_query_conditioning"}

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    early = EarlyStopping(patience=(args.patience if not args.smoke else 1))
    loss_fn = classification_loss
    LOGGER.info(
        "ablation=%s fold=%d params=%d device=%s queries=%d",
        args.ablation,
        args.fold,
        total_params,
        device,
        len(train_frame),
    )

    def score_query(position: int, frame: pd.DataFrame) -> torch.Tensor:
        q = _query_inputs(
            frame, position, inputs["hpo_to_local"], inputs["gene_id_to_local"],
            inputs["n_gene"], inputs["rwr_by_disease"],
        )
        q_d, _ = encoder(q["hpo_indices"].to(device), q["hpo_weights"].to(device))
        hidden = backbone(
            graph, q_d, q["seed_hpo"], inputs["hpo_offset"],
            condition=condition, seed_injection=seed_injection,
        )
        gene_hidden = hidden[torch.tensor(inputs["gene_node_indices"], dtype=torch.long)]
        query_vector = q_d if decoder_query_on else torch.zeros_like(q_d)
        logits = decoder(gene_hidden, query_vector, q["rwr_feature"].to(device))
        weight = torch.ones_like(q["targets"])
        if args.pu:
            row2 = frame.iloc[position]
            positives = {
                inputs["gene_id_to_local"][str(g)]
                for g in row2.covered_positive_gene_ids
                if str(g) in inputs["gene_id_to_local"]
            }
            samples, _ = sample_unlabelled(
                member=0,
                base_seed=derive_seed(args.seed, position),
                positive_genes=positives,
                gene_degrees=inputs["gene_degrees"],
                ratio=args.ratio,
            )
            _, weight = targets_from_sample(inputs["n_gene"], positives, samples)
            if args.loss_focus == "positive_only":
                weight = (q["targets"] > 0).float()
            weight = torch.tensor(weight, dtype=torch.float32)
        return logits, q["targets"].to(device), weight.to(device)

    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        for position in range(len(train_frame)):
            optimizer.zero_grad()
            logits, targets, weight = score_query(position, train_frame)
            loss = loss_fn(logits, targets, sample_weight=weight)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += float(loss.detach())
        model.eval()
        val_total = 0.0
        val_count = min(len(val_frame), 20)
        with torch.no_grad():
            for position in range(val_count):
                logits, targets, weight = score_query(position, val_frame)
                val_total += float(loss_fn(logits, targets, sample_weight=weight))
        val_loss = val_total / max(1, val_count)
        history.append(
            {"epoch": epoch, "train_loss": total / len(train_frame), "val_loss": val_loss}
        )
        stop = early.update({"val_loss": val_loss}, epoch)
        LOGGER.info("epoch=%d train=%.4f val=%.4f best=%d", epoch, history[-1]["train_loss"], val_loss, early.best_epoch)
        if stop:
            break

    output_dir = args.output_dir or (root / "outputs" / "v1" / "conditioned" / args.ablation / f"fold_{args.fold}")
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "model.pt"
    save_checkpoint(
        path=checkpoint_path,
        model=model,
        optimizer=optimizer,
        epoch=early.best_epoch,
        seed=args.seed,
        config_hash=hash_mapping(vars(args)),
        data_hashes={"queries": "fold_%d" % args.fold},
        git_commit=git_commit(root),
        framework_versions={"torch": torch.__version__},
        extra={
            "model": "phenotype_conditioned_rgcn",
            "ablation": args.ablation,
            "fold": args.fold,
            "params": total_params,
            "history": history,
        },
    )
    summary = {
        "pu": args.pu,
        "ratio": args.ratio,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "patience": args.patience,
        "ablation": args.ablation,
        "fold": args.fold,
        "params": total_params,
        "best_epoch": early.best_epoch,
        "best_val_loss": early.best,
        "history": history,
        "checkpoint": str(checkpoint_path),
        "device": str(device),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    LOGGER.info("summary written under %s", output_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
