"""Run the fold-specific Resnik/BMA semantic baseline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from phenotype_network_v0.config import load_config
from phenotype_network_v0.logging_utils import configure_logging
from phenotype_network_v0.models.semantic_baseline import (
    SemanticSimilarity,
    disease_similarities,
    phenotype_sets,
    rank_semantic_query,
    train_gene_disease_sets,
)


def _read_ids(path: Path) -> set[str]:
    return {line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}


def main() -> None:
    """Run semantic ranking for every test fold using streaming Parquet writes."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(__file__).resolve().parents[1] / "configs/default.yaml")
    parser.add_argument("--fold", type=int, help="Run one fold only")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()
    logger = configure_logging(level=args.log_level, logger_name=__name__)
    config = load_config(args.config)
    processed, folds, outputs = (Path(config["paths"][key]) for key in ("processed", "folds", "outputs"))
    phenotypes = pd.read_parquet(processed / "cardiovascular_disease_phenotypes.parquet")
    genes = pd.read_parquet(processed / "cardiovascular_disease_genes.parquet")
    hpo_edges = pd.read_parquet(processed / "hpo_edges.parquet")
    candidates = tuple(pd.read_parquet(folds / "candidate_gene_universe.parquet")["gene_id"].astype(str))
    fold_indices = [args.fold] if args.fold is not None else list(range(int(config["number_of_folds"])))
    if any(index < 0 or index >= int(config["number_of_folds"]) for index in fold_indices):
        raise ValueError("--fold is outside the configured range")
    for fold_index in fold_indices:
        fold_dir = folds / f"fold_{fold_index}"
        train = _read_ids(fold_dir / "train_diseases.txt")
        test = _read_ids(fold_dir / "test_diseases.txt")
        model = SemanticSimilarity(hpo_edges, pd.read_parquet(fold_dir / "train_hpo_ic.parquet"))
        train_phenotypes = phenotype_sets(phenotypes, train)
        test_phenotypes = phenotype_sets(phenotypes, test)
        gene_diseases = train_gene_disease_sets(genes, train)
        output_dir = outputs / "rankings" / f"fold_{fold_index}" / "semantic"
        rankings_path = output_dir / "rankings.parquet"
        similarities_path = output_dir / "disease_similarities.parquet"
        qc_path = output_dir / "semantic_qc.json"
        existing = [path for path in (rankings_path, similarities_path, qc_path) if path.exists()]
        if existing and not args.overwrite:
            raise FileExistsError("Semantic outputs exist; use --overwrite: " + ", ".join(map(str, existing)))
        output_dir.mkdir(parents=True, exist_ok=True)
        temp = rankings_path.with_suffix(".parquet.tmp")
        if temp.exists():
            temp.unlink()
        writer = None
        similarity_rows = []
        total_rows = 0
        try:
            for query_index, disease_id in enumerate(sorted(test)):
                similarities = disease_similarities(test_phenotypes[disease_id], train_phenotypes, model)
                ranking = rank_semantic_query(disease_id, similarities, gene_diseases, candidates)
                ranking.insert(1, "split", "test")
                table = pa.Table.from_pandas(ranking, preserve_index=False)
                if writer is None:
                    writer = pq.ParquetWriter(temp, table.schema, compression="zstd")
                writer.write_table(table)
                total_rows += len(ranking)
                similarity_rows.extend({"query_disease_id": disease_id, "train_disease_id": train_id,
                                        "bma_similarity": score}
                                       for train_id, score in similarities.items())
                if (query_index + 1) % 10 == 0:
                    logger.info("Fold %d semantic queries: %d/%d", fold_index, query_index + 1, len(test))
        finally:
            if writer is not None:
                writer.close()
        temp.replace(rankings_path)
        pd.DataFrame.from_records(similarity_rows).to_parquet(similarities_path, index=False)
        qc = {"fold": fold_index, "train_diseases": len(train), "test_diseases": len(test),
              "candidate_genes": len(candidates), "aggregation_methods": 3, "ranking_rows": total_rows,
              "expected_ranking_rows": len(test) * len(candidates) * 3,
              "cached_term_pairs": model.cached_term_pairs,
              "train_only_labels": True}
        qc_path.write_text(json.dumps(qc, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        logger.info("Fold %d semantic baseline complete: rows=%d", fold_index, total_rows)


if __name__ == "__main__":
    main()