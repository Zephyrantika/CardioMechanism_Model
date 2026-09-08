"""Fold-specific Resnik/BMA semantic similarity and gene ranking."""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

AGGREGATION_METHODS = ("max", "top5_mean", "weighted_sum")


@dataclass
class SemanticSimilarity:
    """Cache-aware Resnik term similarity over an HPO DAG."""

    hpo_edges: pd.DataFrame
    hpo_ic: pd.DataFrame
    _parents: dict[str, set[str]] = field(init=False, repr=False)
    _ancestors_cache: dict[str, set[str]] = field(default_factory=dict, init=False, repr=False)
    _term_similarity_cache: dict[tuple[str, str], float] = field(default_factory=dict, init=False, repr=False)
    _ancestor_mask_cache: dict[str, int] = field(default_factory=dict, init=False, repr=False)
    _bit_position: dict[str, int] = field(init=False, repr=False)
    _bit_ic: tuple[float, ...] = field(init=False, repr=False)
    _ic: dict[str, float] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        parents: dict[str, set[str]] = defaultdict(set)
        for row in self.hpo_edges.itertuples(index=False):
            parents[str(row.child_hpo_id)].add(str(row.parent_hpo_id))
        self._parents = parents
        self._ic = dict(zip(self.hpo_ic["hpo_id"].astype(str), self.hpo_ic["ic"].astype(float)))
        ordered = sorted(self._ic, key=lambda term: (-self._ic[term], term))
        self._bit_position = {term: index for index, term in enumerate(ordered)}
        self._bit_ic = tuple(self._ic[term] for term in ordered)

    def ancestors(self, term_id: str) -> set[str]:
        """Return the term and all transitive ancestors."""
        if term_id not in self._ic:
            raise ValueError(f"HPO term absent from fold IC table: {term_id}")
        if term_id in self._ancestors_cache:
            return self._ancestors_cache[term_id]
        found = {term_id}
        queue = deque([term_id])
        while queue:
            for parent in self._parents.get(queue.popleft(), set()):
                if parent not in found:
                    found.add(parent)
                    queue.append(parent)
        self._ancestors_cache[term_id] = found
        return found

    def ancestor_mask(self, term_id: str) -> int:
        """Encode all ancestors as an IC-ordered integer bit set."""
        if term_id not in self._ancestor_mask_cache:
            mask = 0
            for ancestor in self.ancestors(term_id):
                mask |= 1 << self._bit_position[ancestor]
            self._ancestor_mask_cache[term_id] = mask
        return self._ancestor_mask_cache[term_id]

    def resnik(self, left: str, right: str) -> float:
        """Return the IC of the most informative common ancestor."""
        left, right = str(left), str(right)
        key = (left, right) if left <= right else (right, left)
        if key not in self._term_similarity_cache:
            common = self.ancestor_mask(key[0]) & self.ancestor_mask(key[1])
            if common:
                first_bit = (common & -common).bit_length() - 1
                self._term_similarity_cache[key] = self._bit_ic[first_bit]
            else:
                self._term_similarity_cache[key] = 0.0
        return self._term_similarity_cache[key]

    def bma(self, left: set[str], right: set[str]) -> float:
        """Return symmetric best-match-average similarity for two non-empty HPO sets."""
        if not left or not right:
            raise ValueError("BMA requires two non-empty HPO sets")
        left_score = sum(max(self.resnik(a, b) for b in right) for a in left) / len(left)
        right_score = sum(max(self.resnik(b, a) for a in left) for b in right) / len(right)
        return 0.5 * (left_score + right_score)

    @property
    def cached_term_pairs(self) -> int:
        """Return the number of cached unordered term pairs."""
        return len(self._term_similarity_cache)


def disease_similarities(
    query_terms: set[str], train_phenotypes: dict[str, set[str]], model: SemanticSimilarity,
) -> dict[str, float]:
    """Score one query against every training disease."""
    return {disease_id: model.bma(query_terms, terms) for disease_id, terms in sorted(train_phenotypes.items())}


def rank_semantic_query(
    query_disease_id: str,
    similarities: dict[str, float],
    train_gene_diseases: dict[str, set[str]],
    candidate_gene_ids: tuple[str, ...],
) -> pd.DataFrame:
    """Create complete deterministic candidate rankings for all registered aggregations."""
    gene_numbers = np.asarray([int(gene) for gene in candidate_gene_ids], dtype=np.int64)
    scores = {method: np.zeros(len(candidate_gene_ids), dtype=float) for method in AGGREGATION_METHODS}
    best_disease: list[str | None] = [None] * len(candidate_gene_ids)
    for index, gene_id in enumerate(candidate_gene_ids):
        associated = sorted((similarities[disease], disease) for disease in train_gene_diseases.get(gene_id, set()))
        if not associated:
            continue
        values = [value for value, _ in associated]
        scores["max"][index] = values[-1]
        scores["top5_mean"][index] = float(np.mean(values[-5:]))
        scores["weighted_sum"][index] = float(np.sum(values))
        best_disease[index] = associated[-1][1]
    frames = []
    for method in AGGREGATION_METHODS:
        order = np.lexsort((gene_numbers, -scores[method]))
        frames.append(pd.DataFrame({
            "disease_id": query_disease_id,
            "gene_id": np.asarray(candidate_gene_ids, dtype=object)[order],
            "semantic_score": scores[method][order],
            "rank": np.arange(1, len(candidate_gene_ids) + 1, dtype=np.int32),
            "aggregation_method": method,
            "best_matching_train_disease": np.asarray(best_disease, dtype=object)[order],
        }))
    return pd.concat(frames, ignore_index=True)


def phenotype_sets(frame: pd.DataFrame, disease_ids: set[str]) -> dict[str, set[str]]:
    """Build unique positive HPO sets for exactly the requested diseases."""
    selected = frame.loc[frame["disease_id"].isin(disease_ids)]
    result = {str(disease): set(group["hpo_id"].astype(str)) for disease, group in selected.groupby("disease_id")}
    missing = disease_ids - set(result)
    if missing:
        raise ValueError(f"Diseases lack query phenotypes: {', '.join(sorted(missing))}")
    return result


def train_gene_disease_sets(frame: pd.DataFrame, train_diseases: set[str]) -> dict[str, set[str]]:
    """Map genes to training diseases only."""
    selected = frame.loc[frame["disease_id"].isin(train_diseases), ["gene_id", "disease_id"]].drop_duplicates()
    result: dict[str, set[str]] = defaultdict(set)
    for row in selected.itertuples(index=False):
        result[str(row.gene_id)].add(str(row.disease_id))
    return dict(result)