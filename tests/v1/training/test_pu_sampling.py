"""PU sampling determinism and audit tests (Milestone 4)."""

from __future__ import annotations

import numpy as np

from phenotype_network_v1.training.pu import (
    degree_decile_bins,
    sample_unlabelled,
    targets_from_sample,
)


def _degrees() -> list[float]:
    rng = np.random.RandomState(0)
    return [float(x) for x in rng.randint(1, 500, size=200)]


def test_sampling_deterministic() -> None:
    degrees = _degrees()
    positives = {0, 1, 2}
    first, _ = sample_unlabelled(member=0, base_seed=7, positive_genes=positives,
                                 gene_degrees=degrees, ratio=20)
    second, _ = sample_unlabelled(member=0, base_seed=7, positive_genes=positives,
                                  gene_degrees=degrees, ratio=20)
    assert [(s.gene_local, s.decile) for s in first] == [(s.gene_local, s.decile) for s in second]


def test_different_seeds_or_members_differ() -> None:
    degrees = _degrees()
    positives = {0, 1, 2}
    a, _ = sample_unlabelled(member=0, base_seed=7, positive_genes=positives, gene_degrees=degrees)
    b, _ = sample_unlabelled(member=1, base_seed=7, positive_genes=positives, gene_degrees=degrees)
    assert {s.gene_local for s in a} != {s.gene_local for s in b}


def test_positives_never_sampled() -> None:
    degrees = _degrees()
    positives = {5, 6, 7}
    samples, _ = sample_unlabelled(member=0, base_seed=3, positive_genes=positives, gene_degrees=degrees, ratio=100)
    assert all(record.gene_local not in positives for record in samples)
    assert all(record.role == "sampled_unlabelled" for record in samples)


def test_ratio_bounds_sample_count() -> None:
    degrees = _degrees()
    positives = {0, 1, 2}
    samples, _ = sample_unlabelled(member=0, base_seed=1, positive_genes=positives,
                                   gene_degrees=degrees, ratio=20)
    assert len(samples) <= len(positives) * 20


def test_stratified_vs_uniform_differ() -> None:
    degrees = _degrees()
    positives = {0, 1, 2}
    stratified, _ = sample_unlabelled(member=0, base_seed=9, positive_genes=positives,
                                      gene_degrees=degrees, mode="degree_stratified")
    uniform, _ = sample_unlabelled(member=0, base_seed=9, positive_genes=positives,
                                   gene_degrees=degrees, mode="uniform")
    assert {s.gene_local for s in stratified} != {s.gene_local for s in uniform}


def test_decile_bins_monotonic() -> None:
    bins = degree_decile_bins(_degrees(), n_deciles=10)
    assert len(bins) == 11
    assert all(bins[i] <= bins[i + 1] for i in range(len(bins) - 1))


def test_targets_mask_unlabelled() -> None:
    target, weight = targets_from_sample(
        n_genes=10, positive_genes={1}, samples=[]
    )
    assert target[1] == 1.0 and weight[1] == 1.0
    assert float(target[9]) == 0.0 and float(weight[9]) == 0.0  # unlabelled, masked
    samples, _ = sample_unlabelled(member=0, base_seed=1, positive_genes={1},
                                   gene_degrees=[1.0] * 10, ratio=2)
    target2, weight2 = targets_from_sample(10, {1}, samples)
    assert all(float(weight2[s.gene_local]) == 1.0 for s in samples)
    assert all(float(target2[s.gene_local]) == 0.0 for s in samples)
