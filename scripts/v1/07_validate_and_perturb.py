"""External validation and robustness for frozen V1 checkpoints (M7).

Scores test queries with a frozen conditioned checkpoint and reports: (a)
external-evidence support enrichment in the top-k candidates (GWAS / GTEx /
GO / BioGRID / STRING-900) and (b) rank stability under preregistered edge
perturbations. No retraining; external sources never alter candidates or
weights.

Usage (server): python scripts/v1/07_validate_and_perturb.py --fold 0 --device cuda
CPU smoke:      python scripts/v1/07_validate_and_perturb.py --fold 0 --smoke
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

from phenotype_network_v1.checkpoint import git_commit
from phenotype_network_v1.evaluation.external import (
    external_support_enrichment,
    load_external_gene_sets,
    record_mapping_rules,
)
from phenotype_network_v1.evaluation.robustness import (
    evaluate_perturbation_robustness,
    perturb_edge_index,
)
from phenotype_network_v1.models.conditioned_rgcn import ConditionedRGCN
from phenotype_network_v1.models.decoder import ScoreDecoder
from phenotype_network_v1.models.gcn import build_unified_from_stores
from phenotype_network_v1.models.phenotype_encoder import QueryPhenotypeEncoder
from phenotype_network_v1.training.sampling import seed_everything
from phenotype_network_v1.training.trainer import load_checkpoint

LOGGER = logging.getLogger("v1.validate_external")


def _configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--top-k", type=int, default=200)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)
    _configure_logging(args.log_level)
    root = Path(__file__).resolve().parents[2]
    seed_everything(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")

    checkpoint_path = root / "outputs" / "v1" / "conditioned" / "full" / f"fold_{args.fold}" / "model.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"missing checkpoint: {checkpoint_path}")
    heterodata = torch.load(root / "data" / "graphs" / "v1" / f"fold_{args.fold}" / "heterodata.pt", weights_only=False)
    graph = build_unified_from_stores(heterodata)
    queries = pd.read_parquet(root / "data" / "processed" / "v1" / f"fold_{args.fold}" / "query_instances.parquet")
    node_map = pd.read_parquet(root / "data" / "graphs" / f"fold_{args.fold}" / "node_map.parquet").sort_values("node_index")
    phenotype_rows = node_map[node_map["node_type"] == "phenotype"].reset_index(drop=True)
    gene_rows = node_map[node_map["node_type"] == "gene"].reset_index(drop=True)
    hpo_to_local = {str(r.node_id): int(i) for i, r in phenotype_rows.iterrows()}
    gene_ids = [str(r.node_id) for r in gene_rows.itertuples(index=False)]
    n_gene = len(gene_rows)
    gene_node_indices = list(range(int(graph["offsets"]["gene"]), int(graph["offsets"]["gene"]) + n_gene))
    hpo_offset = int(graph["offsets"]["phenotype"])

    encoder = QueryPhenotypeEncoder(int(graph["sizes"]["phenotype"]), 128, seed=args.seed)
    backbone = ConditionedRGCN(int(graph["num_nodes"]), int(graph["num_relations"]),
                               hidden_dim=128, num_layers=2, relation_bases=8, seed=args.seed)
    decoder = ScoreDecoder(128, seed=args.seed)
    parts = torch.nn.ModuleList([encoder, backbone, decoder]).to(device)
    parts.load_state_dict(load_checkpoint(checkpoint_path)["model_state_dict"])
    parts.eval()

    test_frame = queries[queries["split"] == "test"].reset_index(drop=True)
    if args.smoke:
        test_frame = test_frame.head(1)
        args.top_k = 50

    external_sets = load_external_gene_sets(root / "data" / "processed")
    per_query: list[dict[str, Any]] = []
    for position in range(len(test_frame)):
        row = test_frame.iloc[position]
        hpo_indices = [hpo_to_local[str(h)] for h in row.hpo_ids if str(h) in hpo_to_local]
        weights = [float(w) for hpo, w in zip(row.hpo_ids, row.hpo_weights) if str(hpo) in hpo_to_local]

        def score_all(edge_override=None) -> torch.Tensor:  # noqa: ANN001
            with torch.no_grad():
                q_d, _ = encoder(torch.tensor(hpo_indices, dtype=torch.long).to(device),
                                 torch.tensor(weights, dtype=torch.float32).to(device))
                if edge_override is not None:
                    saved_index, saved_type = graph["edge_index"], graph["edge_type"]
                    graph["edge_index"], graph["edge_type"] = edge_override
                    try:
                        hidden = backbone(graph, q_d, hpo_indices, hpo_offset)
                    finally:
                        graph["edge_index"], graph["edge_type"] = saved_index, saved_type
                else:
                    hidden = backbone(graph, q_d, hpo_indices, hpo_offset)
                gene_hidden = hidden[torch.tensor(gene_node_indices, dtype=torch.long).to(device)]
                logits = decoder(gene_hidden, q_d, torch.zeros(n_gene).to(device))
            return torch.sigmoid(logits)

        scores = score_all().detach().cpu()
        order = scores.argsort(descending=True).tolist()
        ranked = [gene_ids[index] for index in order]
        enrichment = external_support_enrichment(
            ranked_gene_ids=ranked, top_k=args.top_k, external_gene_sets=external_sets
        )
        seeds = [1, 2, 3] if not args.smoke else [1]
        robustness = evaluate_perturbation_robustness(
            score_fn=lambda: score_all().tolist(),
            perturbed_score_fn=lambda seed: score_all(
                perturb_edge_index(
                    graph["edge_index"],
                    0.05 if not args.smoke else 0.01,
                    seed=seed,
                    edge_type=graph["edge_type"],
                )
            ).tolist(),
            seeds=seeds,
        )
        per_query.append(
            {
                "disease_id": str(row.disease_id),
                "external_enrichment_top_k": enrichment,
                "robustness": robustness,
            }
        )
        LOGGER.info("query %s evaluated (%d seeds)", row.disease_id, len(seeds))

    output_dir = args.output_dir or (root / "outputs" / "v1" / "external" / f"fold_{args.fold}")
    output_dir.mkdir(parents=True, exist_ok=True)
    qc = {
        "fold": args.fold,
        "top_k": args.top_k,
        "queries": int(len(test_frame)),
        "mapping_rules": record_mapping_rules(),
        "external_never_alters_candidates_or_weights": True,
        "per_query": per_query,
        "git_commit": git_commit(root),
    }
    (output_dir / "external_validation_qc.json").write_text(
        json.dumps(qc, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    LOGGER.info("external validation complete for fold %d", args.fold)
    return 0


if __name__ == "__main__":
    sys.exit(main())
