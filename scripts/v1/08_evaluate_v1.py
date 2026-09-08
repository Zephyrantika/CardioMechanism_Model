"""Final V1 evaluation vs V0 and deep baselines (Milestone 8).

Scores test queries with frozen V1 checkpoints (ensemble mean when present,
otherwise the conditioned checkpoint), ranks the frozen candidate universe,
and reports Recall@k / MRR next to the V0 raw-RWR comparator computed from
the same frozen rankings. Bootstrap deltas are reported for paired queries.

The frozen V0 evaluator is not replaced; test labels are read only at this
final stage by the evaluator.

Usage (server): python scripts/v1/08_evaluate_v1.py --fold 0..4 --device cuda
CPU smoke:      python scripts/v1/08_evaluate_v1.py --fold 0 --smoke
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
from phenotype_network_v1.evaluation.metrics import compute_metrics
from phenotype_network_v1.evaluation.statistics import paired_bootstrap_delta
from phenotype_network_v1.models.conditioned_rgcn import ConditionedRGCN
from phenotype_network_v1.models.decoder import ScoreDecoder
from phenotype_network_v1.models.gcn import build_unified_from_stores
from phenotype_network_v1.models.phenotype_encoder import QueryPhenotypeEncoder
from phenotype_network_v1.training.sampling import seed_everything

LOGGER = logging.getLogger("v1.evaluate")


def _configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )


def _build_scorer(root: Path, fold: int, device, seed: int):  # noqa: ANN001
    ensemble_dir = root / "outputs" / "v1" / "ensemble" / "degree_stratified" / f"fold_{fold}"
    conditioned_dir = root / "outputs" / "v1" / "conditioned" / "full" / f"fold_{fold}"
    checkpoint_candidates = sorted(ensemble_dir.glob("member_*.pt")) or [None]
    if checkpoint_candidates == [None] and (conditioned_dir / "model.pt").exists():
        checkpoint_candidates = [conditioned_dir / "model.pt"]
        mode = "conditioned"
    elif checkpoint_candidates != [None]:
        mode = "ensemble"
    else:
        return None
    heterodata = torch.load(
        root / "data" / "graphs" / "v1" / f"fold_{fold}" / "heterodata.pt",
        weights_only=False,
    )
    graph = build_unified_from_stores(heterodata)
    node_map = pd.read_parquet(
        root / "data" / "graphs" / f"fold_{fold}" / "node_map.parquet"
    ).sort_values("node_index")
    phenotype_rows = node_map[node_map["node_type"] == "phenotype"].reset_index(drop=True)
    gene_rows = node_map[node_map["node_type"] == "gene"].reset_index(drop=True)
    return {
        "mode": mode,
        "paths": checkpoint_candidates,
        "graph": graph,
        "hpo_to_local": {str(r.node_id): int(i) for i, r in phenotype_rows.iterrows()},
        "gene_ids": [str(r.node_id) for r in gene_rows.itertuples(index=False)],
        "gene_node_indices": list(
            range(int(graph["offsets"]["gene"]), int(graph["offsets"]["gene"]) + len(gene_rows))
        ),
        "hpo_offset": int(graph["offsets"]["phenotype"]),
        "n_gene": len(gene_rows),
        "seed": seed,
        "device": device,
    }


def _ensemble_score(scorer, parts_list, row) -> list[float]:  # noqa: ANN001
    device = scorer["device"]
    hpo_indices = [scorer["hpo_to_local"][str(h)] for h in row.hpo_ids if str(h) in scorer["hpo_to_local"]]
    weights = [float(w) for hpo, w in zip(row.hpo_ids, row.hpo_weights) if str(hpo) in scorer["hpo_to_local"]]
    accumulated = None
    with torch.no_grad():
        for encoder, backbone, decoder in parts_list:
            q_d, _ = encoder(torch.tensor(hpo_indices, dtype=torch.long).to(device),
                             torch.tensor(weights, dtype=torch.float32).to(device))
            hidden = backbone(scorer["graph"], q_d, hpo_indices, scorer["hpo_offset"])
            gene_hidden = hidden[torch.tensor(scorer["gene_node_indices"], dtype=torch.long).to(device)]
            logits = decoder(gene_hidden, q_d, torch.zeros(scorer["n_gene"]).to(device))
            scores = torch.sigmoid(logits)
            accumulated = scores if accumulated is None else accumulated + scores
    return [float(x) for x in (accumulated / len(parts_list)).cpu()]


def _v0_rwr_rankings(root: Path, fold: int) -> dict[str, list[str]]:
    path = root / "outputs" / "rankings" / f"fold_{fold}" / "rwr" / "rankings.parquet"
    if not path.exists():
        return {}
    table = pd.read_parquet(path)
    rankings: dict[str, list[str]] = {}
    for disease_id, frame in table.groupby("disease_id"):
        rankings[str(disease_id)] = frame.sort_values("rank")["gene_id"].astype(str).tolist()
    return rankings


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)
    _configure_logging(args.log_level)
    root = Path(__file__).resolve().parents[2]
    seed_everything(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")

    scorer = _build_scorer(root, args.fold, device, args.seed)
    queries = pd.read_parquet(
        root / "data" / "processed" / "v1" / f"fold_{args.fold}" / "query_instances.parquet"
    )
    test_frame = queries[queries["split"] == "test"].reset_index(drop=True)
    if args.smoke:
        test_frame = test_frame.head(3)
    output_dir = args.output_dir or (root / "outputs" / "v1" / "metrics" / f"fold_{args.fold}")

    rows: list[dict[str, Any]] = []
    v0_rankings = _v0_rwr_rankings(root, args.fold)
    parts_list = []
    if scorer is not None:
        for path in scorer["paths"]:
            payload = load_checkpoint(path)
            hidden_dim = int(payload["model_state_dict"]["1.bases.0"].shape[0])
            parts = torch.nn.ModuleList([
                QueryPhenotypeEncoder(int(scorer["graph"]["sizes"]["phenotype"]), hidden_dim, seed=args.seed),
                ConditionedRGCN(int(scorer["graph"]["num_nodes"]), int(scorer["graph"]["num_relations"]),
                                hidden_dim=hidden_dim, num_layers=2, relation_bases=8, seed=args.seed),
                ScoreDecoder(hidden_dim, seed=args.seed),
            ]).to(device)
            parts.load_state_dict(payload["model_state_dict"])
            parts.eval()
            parts_list.append(parts)

    for position in range(len(test_frame)):
        row = test_frame.iloc[position]
        positives = [str(g) for g in row.covered_positive_gene_ids]
        record: dict[str, Any] = {"disease_id": str(row.disease_id)}
        if scorer is not None and parts_list:
            scores = _ensemble_score(scorer, parts_list, row)
            ranked = [scorer["gene_ids"][i] for i in np.argsort(-np.asarray(scores)).tolist()]
            record["v1_conditioned"] = compute_metrics(ranked, positives)
        v0_ranked = v0_rankings.get(str(row.disease_id))
        if v0_ranked:
            record["v0_rwr"] = compute_metrics(v0_ranked, positives)
        rows.append(record)

    output_dir.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {
        "fold": args.fold,
        "scorer_mode": scorer["mode"] if scorer else "none_pending_server",
        "queries": len(rows),
        "per_query": rows,
        "git_commit": git_commit(root),
    }
    metric_keys = [key for key in ("v1_conditioned", "v0_rwr") if any(key in r for r in rows)]
    for key in metric_keys:
        values = [r[key] for r in rows if key in r]
        summary[f"{key}_aggregate"] = {
            metric: float(np.nanmean([v[metric] for v in values]))
            for metric in ("recall_k10", "recall_k20", "mrr")
        }
    if "v1_conditioned" in metric_keys and "v0_rwr" in metric_keys:
        paired = [(r["v1_conditioned"], r["v0_rwr"]) for r in rows if "v1_conditioned" in r and "v0_rwr" in r]
        summary["v1_vs_v0_bootstrap_mrr_delta"] = paired_bootstrap_delta(
            [a["mrr"] for a, _ in paired], [b["mrr"] for _, b in paired], seed=args.seed,
            iterations=200 if args.smoke else 1000,
        )
    (output_dir / "metrics.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    LOGGER.info("evaluation complete fold %d: %s", args.fold, summary["scorer_mode"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
