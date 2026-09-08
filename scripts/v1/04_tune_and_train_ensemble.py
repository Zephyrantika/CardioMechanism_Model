"""Tune (validation-only) and train the V1 PU ensemble (Milestone 4).

Pipeline per fold:
1. validation-only successive-halving search (access-logged; test never read);
2. deterministic degree-stratified bagging PU training of ``members`` models;
3. auditable sampled-unlabelled records (``label_role``) and per-member seeds.

Uniform random-negative sampling is available only as ``--sampling uniform``
and is always labelled non-primary in the summary.

Usage (server): python scripts/v1/04_tune_and_train_ensemble.py --fold 0 --device cuda
CPU smoke:      python scripts/v1/04_tune_and_train_ensemble.py --fold 0 --smoke
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from phenotype_network_v1.checkpoint import git_commit
from phenotype_network_v1.evaluation.bias import compute_degree_bias
from phenotype_network_v1.models.conditioned_rgcn import ConditionedRGCN
from phenotype_network_v1.models.decoder import ScoreDecoder
from phenotype_network_v1.models.gcn import build_unified_from_stores
from phenotype_network_v1.models.losses import classification_loss
from phenotype_network_v1.models.phenotype_encoder import QueryPhenotypeEncoder
from phenotype_network_v1.training.pu import (
    PuSample,
    sample_unlabelled,
    targets_from_sample,
)
from phenotype_network_v1.training.sampling import derive_seed, seed_everything
from phenotype_network_v1.training.search import (
    AccessLog,
    successive_halving,
)
from phenotype_network_v1.training.trainer import (
    EarlyStopping,
    describe_model,
    save_checkpoint,
)

LOGGER = logging.getLogger("v1.train_ensemble")


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
    # Gene degrees from the frozen graph (in-degree of the gene nodes).
    edge_index = graph["edge_index"]
    degrees = torch.zeros(int(graph["num_nodes"]), dtype=torch.long)
    degrees.index_add_(
        0, edge_index[1], torch.ones(edge_index.size(1), dtype=torch.long)
    )
    gene_degrees = [int(degrees[global_index]) for global_index in gene_node_indices]
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
        "gene_degrees": gene_degrees,
        "hpo_offset": int(graph["offsets"]["phenotype"]),
        "rwr_by_disease": rwr_by_disease,
    }


def _query_row(frame: pd.DataFrame, position: int):
    return frame.iloc[position]


def _build_pu_labels(
    inputs: dict[str, Any],
    frame: pd.DataFrame,
    position: int,
    *,
    member: int,
    base_seed: int,
    ratio: int,
    sampling: str,
) -> tuple[np.ndarray, np.ndarray, list[PuSample]]:
    row = _query_row(frame, position)
    gene_id_to_local = inputs["gene_id_to_local"]
    positives = {
        gene_id_to_local[str(gene_id)]
        for gene_id in row.covered_positive_gene_ids
        if str(gene_id) in gene_id_to_local
    }
    samples, _counts = sample_unlabelled(
        member=member,
        base_seed=derive_seed(base_seed, position),
        positive_genes=positives,
        gene_degrees=inputs["gene_degrees"],
        ratio=ratio,
        mode=sampling,
    )
    target, weight = targets_from_sample(inputs["n_gene"], positives, samples)
    return target, weight, samples


def _score_logits(inputs, model, encoder, backbone, decoder, row, device):
    hpo_indices = [
        inputs["hpo_to_local"][str(hpo)]
        for hpo in row.hpo_ids
        if str(hpo) in inputs["hpo_to_local"]
    ]
    weights = []
    for hpo, weight in zip(row.hpo_ids, row.hpo_weights):
        if str(hpo) in inputs["hpo_to_local"]:
            weights.append(float(weight))
    q_d, _ = encoder(
        torch.tensor(hpo_indices, dtype=torch.long).to(device),
        torch.tensor(weights, dtype=torch.float32).to(device),
    )
    hidden = backbone(inputs["graph"], q_d, hpo_indices, inputs["hpo_offset"])
    gene_hidden = hidden[
        torch.tensor(inputs["gene_node_indices"], dtype=torch.long).to(device)
    ]
    rwr_feature = torch.zeros(inputs["n_gene"])
    for gene_id, score in inputs["rwr_by_disease"].get(str(row.disease_id), {}).items():
        local = inputs["gene_id_to_local"].get(str(gene_id))
        if local is not None:
            rwr_feature[local] = float(np.log1p(score))
    return decoder(gene_hidden, q_d, rwr_feature.to(device))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--members", type=int, default=5)
    parser.add_argument("--ratio", type=int, default=20)
    parser.add_argument("--sampling", choices=("degree_stratified", "uniform"), default="degree_stratified")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)
    _configure_logging(args.log_level)
    root = Path(__file__).resolve().parents[2]
    seed_everything(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    inputs = load_fold_inputs(root, args.fold)
    queries = inputs["queries"]
    train_frame = queries[queries["split"] == "train"]
    val_frame = queries[queries["split"] == "validation"]
    if args.smoke:
        train_frame = train_frame.head(3)
        val_frame = val_frame.head(3)
        args.members = min(args.members, 2)

    # --- validation-only tuning with access log ---
    def objective(config: dict[str, Any], access_log: AccessLog) -> float:
        access_log.record("validation_query_instances")
        torch.manual_seed(args.seed)
        hidden_dim = int(config["hidden_dim"])
        encoder = QueryPhenotypeEncoder(int(inputs["graph"]["sizes"]["phenotype"]), hidden_dim, seed=args.seed)
        backbone = ConditionedRGCN(int(inputs["graph"]["num_nodes"]), int(inputs["graph"]["num_relations"]),
                                   hidden_dim=hidden_dim, num_layers=2, relation_bases=8,
                                   rho=float(config["rho"]), seed=args.seed)
        decoder = ScoreDecoder(hidden_dim, seed=args.seed)
        parts = torch.nn.ModuleList([encoder, backbone, decoder]).to(device)
        optimizer = torch.optim.AdamW(parts.parameters(), lr=float(config["lr"]), weight_decay=1e-5)
        loss_fn = classification_loss
        for epoch in range(2 if args.smoke else 8):
            parts.train()
            for position in range(len(train_frame)):
                optimizer.zero_grad()
                row = _query_row(train_frame, position)
                logits = _score_logits(inputs, parts, encoder, backbone, decoder, row, device)
                target, weight, _ = _build_pu_labels(inputs, train_frame, position, member=0,
                                                     base_seed=args.seed, ratio=args.ratio,
                                                     sampling=args.sampling)
                loss = loss_fn(logits, torch.tensor(target).to(device),
                               sample_weight=torch.tensor(weight).to(device))
                loss.backward()
                optimizer.step()
        parts.eval()
        total = 0.0
        with torch.no_grad():
            for position in range(min(len(val_frame), 6)):
                row = _query_row(val_frame, position)
                logits = _score_logits(inputs, parts, encoder, backbone, decoder, row, device)
                target, weight, _ = _build_pu_labels(inputs, val_frame, position, member=0,
                                                     base_seed=args.seed, ratio=args.ratio,
                                                     sampling=args.sampling)
                total += float(loss_fn(logits, torch.tensor(target).to(device),
                                       sample_weight=torch.tensor(weight).to(device)))
        return total / max(1, min(len(val_frame), 6))

    search = successive_halving(
        space={"lr": [3e-4, 1e-3] if not args.smoke else [1e-3],
               "rho": [0.1, 0.2] if not args.smoke else [0.2],
               "hidden_dim": [128] if not args.smoke else [64]},
        objective=objective,
        seed=args.seed,
    )
    LOGGER.info("best config=%s value=%.4f", search.best_config, search.best_value)

    # --- deterministic degree-stratified bagging ensemble ---
    audit_rows: list[dict[str, Any]] = []
    member_summaries = []
    for member in range(args.members):
        member_seed = derive_seed(args.seed, member)
        seed_everything(member_seed)
        hidden_dim = int(search.best_config["hidden_dim"])
        encoder = QueryPhenotypeEncoder(int(inputs["graph"]["sizes"]["phenotype"]), hidden_dim, seed=member_seed)
        backbone = ConditionedRGCN(int(inputs["graph"]["num_nodes"]), int(inputs["graph"]["num_relations"]),
                                   hidden_dim=hidden_dim, num_layers=2, relation_bases=8,
                                   rho=float(search.best_config["rho"]), seed=member_seed)
        decoder = ScoreDecoder(hidden_dim, seed=member_seed)
        parts = torch.nn.ModuleList([encoder, backbone, decoder]).to(device)
        optimizer = torch.optim.AdamW(parts.parameters(), lr=float(search.best_config["lr"]), weight_decay=1e-5)
        loss_fn = classification_loss
        early = EarlyStopping(patience=3 if args.smoke else 30)
        for epoch in range(1, (3 if args.smoke else 300) + 1):
            parts.train()
            for position in range(len(train_frame)):
                optimizer.zero_grad()
                row = _query_row(train_frame, position)
                logits = _score_logits(inputs, parts, encoder, backbone, decoder, row, device)
                target, weight, samples = _build_pu_labels(
                    inputs, train_frame, position, member=member,
                    base_seed=args.seed, ratio=args.ratio, sampling=args.sampling)
                audit_rows.extend(
                    {
                        "member": record.member,
                        "gene_local": record.gene_local,
                        "decile": record.decile,
                        "label_role": record.role,
                        "seed": derive_seed(args.seed, position),
                    }
                    for record in samples
                )
                loss = loss_fn(logits, torch.tensor(target).to(device),
                               sample_weight=torch.tensor(weight).to(device))
                loss.backward()
                optimizer.step()
            parts.eval()
            val_total = 0.0
            with torch.no_grad():
                for position in range(min(len(val_frame), 6)):
                    row = _query_row(val_frame, position)
                    logits = _score_logits(inputs, parts, encoder, backbone, decoder, row, device)
                    target, weight, _ = _build_pu_labels(inputs, val_frame, position, member=member,
                                                         base_seed=args.seed, ratio=args.ratio,
                                                         sampling=args.sampling)
                    val_total += float(loss_fn(logits, torch.tensor(target).to(device),
                                               sample_weight=torch.tensor(weight).to(device)))
            val_loss = val_total / max(1, min(len(val_frame), 6))
            if early.update({"val_loss": val_loss}, epoch):
                break
        output_dir = args.output_dir or (root / "outputs" / "v1" / "ensemble" / args.sampling / f"fold_{args.fold}")
        output_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = output_dir / f"member_{member}.pt"
        save_checkpoint(
            path=checkpoint_path, model=parts, optimizer=optimizer,
            epoch=early.best_epoch, seed=member_seed,
            config_hash=json.dumps({**search.best_config, "members": args.members, "ratio": args.ratio, "sampling": args.sampling}, sort_keys=True),
            data_hashes={"fold": str(args.fold)}, git_commit=git_commit(root),
            framework_versions={"torch": torch.__version__},
            extra={"member": member, "role": "ensemble", "best_val_loss": early.best},
        )
        member_summaries.append(
            {"member": member, "seed": member_seed, "params": sum(describe_model(p) for p in parts),
             "best_epoch": early.best_epoch, "best_val_loss": early.best}
        )
        LOGGER.info("member %d done: best_val=%.4f", member, early.best or float("nan"))

    audit = pd.DataFrame(audit_rows)
    output_dir = args.output_dir or (root / "outputs" / "v1" / "ensemble" / args.sampling / f"fold_{args.fold}")
    audit_path = output_dir / "pu_audit.parquet"
    audit.to_parquet(audit_path, index=False)
    summary = {
        "fold": args.fold,
        "sampling": args.sampling,
        "primary": args.sampling == "degree_stratified",
        "members": args.members,
        "ratio": args.ratio,
        "search": {"best_config": search.best_config, "best_value": search.best_value,
                   "access_log": search.access_log.to_dict()},
        "members_summary": member_summaries,
        "audit_rows": len(audit),
        "audit_roles": audit["label_role"].value_counts().to_dict() if len(audit) else {},
        "git_commit": git_commit(root),
    }
    (output_dir / "ensemble_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    LOGGER.info(
        "ensemble complete: members=%d primary=%s audit_rows=%d",
        args.members, summary["primary"], len(audit),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
