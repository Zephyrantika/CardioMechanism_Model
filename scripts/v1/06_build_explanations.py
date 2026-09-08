"""Build learned path explanations with counterfactual fidelity (M6).

For each test query, reasons bounded paths from HPO seeds to covered positive
genes over the frozen fold graph (approved relations only), scores them with
the trained conditioned model, and measures counterfactual edge-deletion
fidelity. Unresolved targets are preserved and reported; held-out labels are
never used to search or score paths.

Usage (server): python scripts/v1/06_build_explanations.py --fold 0 --device cuda
CPU smoke:      python scripts/v1/06_build_explanations.py --fold 0 --smoke
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
from phenotype_network_v1.evaluation.explanation_fidelity import counterfactual_fidelity
from phenotype_network_v1.models.conditioned_rgcn import ConditionedRGCN
from phenotype_network_v1.models.decoder import ScoreDecoder
from phenotype_network_v1.models.gcn import build_unified_from_stores
from phenotype_network_v1.models.path_reasoner import build_adjacency, reason_paths
from phenotype_network_v1.models.phenotype_encoder import QueryPhenotypeEncoder
from phenotype_network_v1.training.sampling import seed_everything
from phenotype_network_v1.training.trainer import load_checkpoint

LOGGER = logging.getLogger("v1.build_explanations")


def _configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )


def _load_inputs(root: Path, fold: int) -> dict[str, Any]:
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
    return {
        "graph": graph,
        "queries": queries,
        "hpo_to_global": {
            str(row.node_id): int(row.node_index) for row in phenotype_rows.itertuples(index=False)
        },
        "gene_id_to_global": {
            str(row.node_id): int(row.node_index) for row in gene_rows.itertuples(index=False)
        },
        "n_gene": len(gene_rows),
        "gene_node_indices": list(
            range(int(graph["offsets"]["gene"]), int(graph["offsets"]["gene"]) + len(gene_rows))
        ),
        "hpo_offset": int(graph["offsets"]["phenotype"]),
        "adjacency": build_adjacency(graph["edge_index"], graph["edge_type"], graph["relation_names"]),
    }


def _model_parts(inputs, hidden_dim: int, seed: int):
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


class GeneScorer:
    """Scores one gene (global index) for a query, with optional edge removal."""

    def __init__(self, parts, inputs, hpo_globals: list[int], device) -> None:  # noqa: ANN001
        self.parts = parts
        self.inputs = inputs
        self.hpo_globals = hpo_globals
        self.device = device
        encoder, backbone, decoder = parts
        hpo_local = [g - inputs["hpo_offset"] for g in hpo_globals]
        weights = [1.0] * len(hpo_local)
        self.q_d, _ = encoder(
            torch.tensor(hpo_local, dtype=torch.long).to(device),
            torch.tensor(weights, dtype=torch.float32).to(device),
        )
        self.gene_index_to_local = {
            int(global_index): local
            for local, global_index in enumerate(inputs["gene_node_indices"])
        }

    def score(self, gene_global: int, without_edge: tuple[int, int] | None = None) -> float:
        encoder, backbone, decoder = self.parts
        edge_index = self.inputs["graph"]["edge_index"]
        if without_edge is not None:
            keep = ~(
                (edge_index[0] == without_edge[0]) & (edge_index[1] == without_edge[1])
            )
            edge_index = edge_index[:, keep]
        with torch.no_grad():
            hidden = backbone(
                self.inputs["graph"], self.q_d, [g - self.inputs["hpo_offset"] for g in self.hpo_globals],
                self.inputs["hpo_offset"],
            )
            local = self.gene_index_to_local[gene_global]
            gene_hidden = hidden[torch.tensor(self.inputs["gene_node_indices"][local], dtype=torch.long).to(self.device)].unsqueeze(0)
            logit = decoder(gene_hidden, self.q_d, torch.zeros(1).to(self.device))
        return float(torch.sigmoid(logit).item())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-length", type=int, default=4)
    parser.add_argument("--max-paths", type=int, default=20)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)
    _configure_logging(args.log_level)
    root = Path(__file__).resolve().parents[2]
    seed_everything(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    inputs = _load_inputs(root, args.fold)
    test_frame = inputs["queries"][inputs["queries"]["split"] == "test"].reset_index(drop=True)
    if args.smoke:
        test_frame = test_frame.head(2)
        args.max_paths = 5
        args.max_length = 3

    checkpoint_path = (
        root / "outputs" / "v1" / "conditioned" / "full" / f"fold_{args.fold}" / "model.pt"
    )
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"missing conditioned checkpoint: {checkpoint_path}")
    parts = _model_parts(inputs, 128, args.seed).to(device)
    parts.load_state_dict(load_checkpoint(checkpoint_path)["model_state_dict"])
    parts.eval()

    rows: list[dict[str, Any]] = []
    unresolved_report: list[dict[str, Any]] = []
    for position in range(len(test_frame)):
        row = test_frame.iloc[position]
        hpo_globals = [
            inputs["hpo_to_global"][str(hpo)]
            for hpo in row.hpo_ids
            if str(hpo) in inputs["hpo_to_global"]
        ]
        covered_globals = [
            inputs["gene_id_to_global"][str(gene_id)]
            for gene_id in row.covered_positive_gene_ids
            if str(gene_id) in inputs["gene_id_to_global"]
        ]
        for gene_id in row.unresolved_positive_gene_ids:
            unresolved_report.append(
                {"disease_id": str(row.disease_id), "gene_id": str(gene_id),
                 "reason": "not_in_candidate_universe"}
            )
        scorer = GeneScorer(parts, inputs, hpo_globals, device)
        target_set = set(covered_globals)
        paths = reason_paths(
            inputs["adjacency"], hpo_globals, target_set,
            max_length=args.max_length, max_paths=args.max_paths,
        )
        query_rows = 0
        for path in paths:
            gene_global = path.nodes[-1]
            gene_local = scorer.gene_index_to_local.get(gene_global)
            if gene_local is None:
                continue
            fidelity = counterfactual_fidelity(
                path_nodes=path.nodes,
                path_relations=path.relations,
                score_gene=lambda: scorer.score(gene_global),
                score_gene_without_edge=lambda u, v: scorer.score(
                    gene_global, without_edge=(u, v)
                ),
            )
            rows.append(
                {
                    "fold": args.fold,
                    "disease_id": str(row.disease_id),
                    "gene_global": int(gene_global),
                    "path_nodes": path.nodes,
                    "path_relations": path.relations,
                    "baseline_score": fidelity["baseline_score"],
                    "fidelity": fidelity["fidelity"],
                    "mean_fidelity": fidelity["mean_fidelity"],
                }
            )
            query_rows += 1
        LOGGER.info(
            "query %s: seeds=%d targets=%d paths=%d unresolved=%d",
            row.disease_id, len(hpo_globals), len(target_set), query_rows,
            len(row.unresolved_positive_gene_ids),
        )

    frame = pd.DataFrame(rows)
    output_dir = args.output_dir or (root / "outputs" / "v1" / "explanations" / f"fold_{args.fold}")
    output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(output_dir / "paths.parquet", index=False)
    pd.DataFrame(unresolved_report).to_parquet(
        output_dir / "unresolved_targets.parquet", index=False
    )
    qc = {
        "fold": args.fold,
        "queries": int(len(test_frame)),
        "paths": int(len(frame)),
        "unresolved_targets": len(unresolved_report),
        "max_length": args.max_length,
        "max_paths": args.max_paths,
        "git_commit": git_commit(root),
    }
    (output_dir / "explanation_qc.json").write_text(
        json.dumps(qc, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    LOGGER.info("explanations complete: paths=%d unresolved=%d", qc["paths"], qc["unresolved_targets"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
