"""Extract query-specific pathway modules with audit and stability (M5).

Loads a trained V1 conditioned checkpoint (or trains briefly when
``--train-with-module-loss``), scores candidate genes per query, activates
Reactome pathways from frozen membership, extracts soft modules, and audits
them (collapsed / single-hub-dominated / unstable / unsupported -> rejected).
The V0 Leiden comparator is untouched.

Usage (server): python scripts/v1/05_train_modules.py --fold 0 --device cuda
CPU smoke:      python scripts/v1/05_train_modules.py --fold 0 --smoke
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
from phenotype_network_v1.training.trainer import load_checkpoint
from phenotype_network_v1.evaluation.module_stability import (
    audit_module,
    extract_modules,
    split_half_stability,
)
from phenotype_network_v1.models.conditioned_rgcn import ConditionedRGCN
from phenotype_network_v1.models.decoder import ScoreDecoder
from phenotype_network_v1.models.gcn import build_unified_from_stores
from phenotype_network_v1.models.losses import classification_loss
from phenotype_network_v1.models.module_losses import module_regularization_loss
from phenotype_network_v1.models.pathway_hypergraph import (
    load_reactome_membership,
    module_top_genes,
    pathway_activation,
)
from phenotype_network_v1.models.phenotype_encoder import QueryPhenotypeEncoder
from phenotype_network_v1.training.sampling import seed_everything

LOGGER = logging.getLogger("v1.train_modules")


def _configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )


def _load_fold_inputs(root: Path, fold: int) -> dict[str, Any]:
    heterodata = torch.load(
        root / "data" / "graphs" / "v1" / f"fold_{fold}" / "heterodata.pt",
        weights_only=False,
    )
    queries = pd.read_parquet(
        root / "data" / "processed" / "v1" / f"fold_{fold}" / "query_instances.parquet"
    )
    node_map = pd.read_parquet(
        root / "data" / "graphs" / f"fold_{fold}" / "node_map.parquet"
    ).sort_values("node_index")
    gene_rows = node_map[node_map["node_type"] == "gene"].reset_index(drop=True)
    phenotype_rows = node_map[node_map["node_type"] == "phenotype"].reset_index(drop=True)
    graph = build_unified_from_stores(heterodata)
    n_gene = len(gene_rows)
    return {
        "graph": graph,
        "queries": queries,
        "hpo_to_local": {str(r.node_id): int(i) for i, r in phenotype_rows.iterrows()},
        "gene_ids": [str(r.node_id) for r in gene_rows.itertuples(index=False)],
        "n_gene": n_gene,
        "gene_node_indices": list(
            range(int(graph["offsets"]["gene"]), int(graph["offsets"]["gene"]) + n_gene)
        ),
        "hpo_offset": int(graph["offsets"]["phenotype"]),
        "membership": load_reactome_membership(
            root / "data" / "processed" / "reactome_gene_pathway.parquet"
        ),
    }


def _model_parts(inputs: dict[str, Any], hidden_dim: int, seed: int):
    encoder = QueryPhenotypeEncoder(
        int(inputs["graph"]["sizes"]["phenotype"]), hidden_dim, seed=seed
    )
    backbone = ConditionedRGCN(
        int(inputs["graph"]["num_nodes"]),
        int(inputs["graph"]["num_relations"]),
        hidden_dim=hidden_dim,
        num_layers=2,
        relation_bases=8,
        seed=seed,
    )
    decoder = ScoreDecoder(hidden_dim, seed=seed)
    return torch.nn.ModuleList([encoder, backbone, decoder])


def _score_query(parts, inputs, row, device) -> tuple[list[str], list[float]]:
    encoder, backbone, decoder = parts
    hpo_indices = [
        inputs["hpo_to_local"][str(hpo)]
        for hpo in row.hpo_ids
        if str(hpo) in inputs["hpo_to_local"]
    ]
    weights = [
        float(w)
        for hpo, w in zip(row.hpo_ids, row.hpo_weights)
        if str(hpo) in inputs["hpo_to_local"]
    ]
    encoder.eval(); backbone.eval(); decoder.eval()
    with torch.no_grad():
        q_d, _ = encoder(
            torch.tensor(hpo_indices, dtype=torch.long).to(device),
            torch.tensor(weights, dtype=torch.float32).to(device),
        )
        hidden = backbone(inputs["graph"], q_d, hpo_indices, inputs["hpo_offset"])
        gene_hidden = hidden[
            torch.tensor(inputs["gene_node_indices"], dtype=torch.long).to(device)
        ]
        logits = decoder(gene_hidden, q_d, torch.zeros(inputs["n_gene"]).to(device))
    scores = [float(x) for x in torch.sigmoid(logits).detach().cpu()]
    return inputs["gene_ids"], scores


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--top-pathways", type=int, default=20)
    parser.add_argument("--min-stability", type=float, default=0.0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--train-with-module-loss", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)
    _configure_logging(args.log_level)
    root = Path(__file__).resolve().parents[2]
    seed_everything(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    inputs = _load_fold_inputs(root, args.fold)
    frame = inputs["queries"]
    test_frame = frame[frame["split"] == "test"].reset_index(drop=True)
    if args.smoke:
        test_frame = test_frame.head(3)
        args.top_pathways = 8

    parts = None
    checkpoint_dir = root / "outputs" / "v1" / "conditioned" / "full" / f"fold_{args.fold}"
    checkpoint_path = checkpoint_dir / "model.pt"
    if args.train_with_module_loss or not checkpoint_path.exists():
        hidden_dim = 64 if args.smoke else 128
        parts = _model_parts(inputs, hidden_dim, args.seed).to(device)
        loss_fn = classification_loss
        optimizer = torch.optim.AdamW(parts.parameters(), lr=1e-3)
        train_frame = frame[frame["split"] == "train"].reset_index(drop=True)
        for epoch in range(2 if args.smoke else 8):
            parts.train()
            total_loss = 0.0
            for position in range(min(len(train_frame), 8 if args.smoke else len(train_frame))):
                optimizer.zero_grad()
                row = train_frame.iloc[position]
                gene_ids, scores = _score_query(parts, inputs, row, device)
                score_by_gene = dict(zip(gene_ids, scores))
                activation = pathway_activation(inputs["membership"], gene_ids, scores)
                module_loss, _ = module_regularization_loss(
                    activation, inputs["membership"], gene_scores_by_gene=score_by_gene
                )
                targets = torch.zeros(inputs["n_gene"])
                for gene_id in row.covered_positive_gene_ids:
                    if gene_id in inputs["gene_ids"]:
                        targets[inputs["gene_ids"].index(gene_id)] = 1.0
                encoder, backbone, decoder = parts
                hpo_indices = [inputs["hpo_to_local"][str(h)] for h in row.hpo_ids if str(h) in inputs["hpo_to_local"]]
                weights = [float(w) for hpo, w in zip(row.hpo_ids, row.hpo_weights) if str(hpo) in inputs["hpo_to_local"]]
                q_d, _ = encoder(torch.tensor(hpo_indices, dtype=torch.long).to(device),
                                 torch.tensor(weights).to(device))
                hidden = backbone(inputs["graph"], q_d, hpo_indices, inputs["hpo_offset"])
                gene_hidden = hidden[torch.tensor(inputs["gene_node_indices"], dtype=torch.long).to(device)]
                logits = decoder(gene_hidden, q_d, torch.zeros(inputs["n_gene"]).to(device))
                loss = loss_fn(logits, targets.to(device)) + 0.1 * module_loss
                loss.backward()
                optimizer.step()
                total_loss += float(loss.detach())
            LOGGER.info("epoch=%d loss=%.4f", epoch, total_loss / max(1, min(len(train_frame), 8 if args.smoke else len(train_frame))))
    else:
        hidden_dim = 128
        parts = _model_parts(inputs, hidden_dim, args.seed).to(device)
        payload = load_checkpoint(checkpoint_path)
        parts.load_state_dict(payload["model_state_dict"])
        LOGGER.info("loaded conditioned checkpoint %s (epoch %s)", checkpoint_path, payload.get("epoch"))

    module_rows: list[dict[str, Any]] = []
    for position in range(len(test_frame)):
        row = test_frame.iloc[position]
        gene_ids, scores = _score_query(parts, inputs, row, device)
        score_by_gene = dict(zip(gene_ids, scores))
        activation = pathway_activation(inputs["membership"], gene_ids, scores)
        top_pathways = sorted(activation, key=activation.get, reverse=True)[: args.top_pathways]
        for pathway_id in top_pathways:
            module = set(module_top_genes(inputs["membership"], pathway_id, gene_ids, scores, max_genes=100))
            stability = _query_stability(module, gene_ids, scores, inputs["membership"], pathway_id, seed=args.seed)
            reasons = audit_module(module, gene_scores=score_by_gene, min_stability=args.min_stability, stability=stability)
            module_rows.append(
                {
                    "fold": args.fold,
                    "disease_id": str(row.disease_id),
                    "pathway_id": pathway_id,
                    "activation": float(activation[pathway_id]),
                    "module_size": len(module),
                    "stability": float(stability),
                    "module_genes": sorted(module),
                    "reasons": reasons,
                    "retained": not reasons,
                }
            )
        LOGGER.info("query %s: pathways=%d", row.disease_id, len(top_pathways))

    modules = pd.DataFrame(module_rows)
    output_dir = args.output_dir or (root / "outputs" / "v1" / "modules" / f"fold_{args.fold}")
    output_dir.mkdir(parents=True, exist_ok=True)
    modules.to_parquet(output_dir / "modules.parquet", index=False)
    reason_counts: dict[str, int] = {}
    rejected_count = 0
    if len(modules):
        rejected = modules[~modules["retained"]]
        rejected.to_parquet(output_dir / "module_audit.parquet", index=False)
        rejected_count = int(len(rejected))
        for reasons in modules["reasons"]:
            for reason in reasons:
                reason_counts[reason] = reason_counts.get(reason, 0) + 1
    else:
        pd.DataFrame().to_parquet(output_dir / "module_audit.parquet", index=False)
    qc = {
        "fold": args.fold,
        "queries": int(len(test_frame)),
        "modules": int(len(modules)),
        "retained": int(len(modules) - rejected_count),
        "rejected": rejected_count,
        "reason_counts": reason_counts,
        "git_commit": git_commit(root),
    }
    (output_dir / "module_qc.json").write_text(
        json.dumps(qc, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    LOGGER.info("module extraction complete: retained=%d rejected=%d", qc["retained"], qc["rejected"])
    return 0


def _query_stability(
    module: set[str],
    gene_ids: list[str],
    scores: list[float],
    membership: dict[str, set[str]],
    pathway_id: str,
    seed: int,
) -> float:
    rng = np.random.RandomState(seed)
    index = rng.permutation(len(gene_ids))
    half = index[: len(index) // 2]
    other = index[len(index) // 2:]
    ids_a = [gene_ids[i] for i in half]
    scores_a = [scores[i] for i in half]
    ids_b = [gene_ids[i] for i in other]
    scores_b = [scores[i] for i in other]
    return split_half_stability(
        {pathway_id: module}, ids_a, scores_a, ids_b, scores_b
    )[pathway_id]


if __name__ == "__main__":
    sys.exit(main())
