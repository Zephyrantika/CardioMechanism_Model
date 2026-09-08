"""Degree-stratified bagging PU sampling for V1 (Milestone 4).

Deterministic positive-unlabelled sampling: five ensemble members by default;
each member samples unlabelled genes within graph-degree deciles at a
sampled-unlabelled to positive ratio of 20:1; every member uses an explicit
derived seed; sampled records are auditable via ``label_role``. Uniform
random-negative sampling is available only as a clearly labelled non-primary
ablation. Unlabelled genes are never treated as confirmed negatives.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from phenotype_network_v1.training.sampling import derive_seed

SAMPLING_MODES = ("degree_stratified", "uniform")


@dataclass(frozen=True)
class PuSample:
    """Audit record of one sampled-unlabelled gene."""

    member: int
    gene_local: int
    decile: int
    role: str = "sampled_unlabelled"


def degree_decile_bins(degrees: Sequence[float], n_deciles: int = 10) -> list[float]:
    """Quantile boundaries partitioning gene degrees into deciles."""
    if n_deciles <= 0:
        raise ValueError("n_deciles must be positive")
    if len(degrees) == 0:
        return []
    quantiles = np.linspace(0.0, 1.0, n_deciles + 1)
    return [float(boundary) for boundary in np.quantile(list(degrees), quantiles)]


def _decile_of(degree: float, bins: list[float]) -> int:
    for index, boundary in enumerate(bins):
        if degree <= boundary:
            return index
    return len(bins) - 1


def sample_unlabelled(
    *,
    member: int,
    base_seed: int,
    positive_genes: set[int],
    gene_degrees: Sequence[float],
    ratio: int = 20,
    mode: str = SAMPLING_MODES[0],
    n_deciles: int = 10,
) -> tuple[list[PuSample], dict[str, int]]:
    """Deterministically sample unlabelled genes for one ensemble member.

    Returns the audit records and a summary of decile counts. Deterministic:
    identical inputs and seed reproduce identical samples.
    """
    if mode not in SAMPLING_MODES:
        raise ValueError(f"mode must be one of {SAMPLING_MODES}")
    if ratio <= 0:
        raise ValueError("ratio must be positive")
    n_genes = len(gene_degrees)
    if n_genes == 0:
        return [], {}
    all_indices = set(range(n_genes))
    unlabelled = sorted(all_indices - positive_genes)
    n_positive = len(positive_genes)
    n_to_sample = min(len(unlabelled), n_positive * ratio)
    rng = np.random.RandomState(derive_seed(base_seed, member * 31 + 7))
    sample_records: list[PuSample] = []
    decile_counts: dict[str, int] = {}

    if mode == "uniform":
        chosen = set(rng.choice(unlabelled, size=n_to_sample, replace=False).tolist())
        for gene in sorted(chosen):
            sample_records.append(
                PuSample(member=member, gene_local=gene, decile=-1)
            )
        decile_counts["uniform"] = len(chosen)
        return sample_records, decile_counts

    bins = degree_decile_bins(gene_degrees, n_deciles)
    deciles_of_gene = {
        gene: _decile_of(float(gene_degrees[gene]), bins) for gene in unlabelled
    }
    # Degree-stratified bagging: per-decile proportional target.
    grouped: dict[int, list[int]] = {}
    for gene in unlabelled:
        grouped.setdefault(deciles_of_gene[gene], []).append(gene)
    remaining = n_to_sample
    for decile in sorted(grouped):
        pool = grouped[decile]
        n_take = max(1, int(np.ceil(remaining * len(pool) / max(1, len(unlabelled)))))
        n_take = min(len(pool), n_take)
        chosen = set(rng.choice(pool, size=n_take, replace=False).tolist())
        for gene in sorted(chosen):
            sample_records.append(
                PuSample(member=member, gene_local=gene, decile=int(decile))
            )
        decile_counts[str(decile)] = len(chosen)
        remaining -= n_take
    return sample_records, decile_counts


def targets_from_sample(
    n_genes: int, positive_genes: set[int], samples: list[PuSample]
) -> tuple[np.ndarray, np.ndarray]:
    """Build (target, weight) arrays for the PU loss.

    Positives get target 1 (weight 1). Sampled unlabelled get target 0 and
    weight 1 so they enter the loss. Every other gene is masked out (weight 0)
    - it is unlabelled, not a confirmed negative.
    """
    target = np.zeros(n_genes, dtype=np.float32)
    weight = np.zeros(n_genes, dtype=np.float32)
    for gene in positive_genes:
        target[gene] = 1.0
        weight[gene] = 1.0
    for record in samples:
        if record.role != "sampled_unlabelled":
            raise ValueError("only sampled_unlabelled records may enter the loss")
        weight[record.gene_local] = 1.0  # target stays 0
    return target, weight


__all__ = [
    "SAMPLING_MODES",
    "PuSample",
    "degree_decile_bins",
    "sample_unlabelled",
    "targets_from_sample",
]
