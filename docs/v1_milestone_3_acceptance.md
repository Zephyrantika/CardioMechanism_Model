# V1 Milestone 3 Acceptance: Phenotype-Conditioned Primary Model

Date: 2026-09-08
Branch: `codex/v1`
Commit: (M3 commit, on top of `03f0b8b`)
Scope: query encoder, FiLM-conditioned basis R-GCN, recurrent seed injection,
RWR-residual decoder, and the five required ablations, CPU-verified.

## 1. Objective

Implement the query encoder, FiLM-conditioned R-GCN, recurrent phenotype seed
injection, and RWR residual decoder defined in Section 7.

## 2. Created files

```text
src/phenotype_network_v1/models/phenotype_encoder.py
src/phenotype_network_v1/models/conditioned_rgcn.py
src/phenotype_network_v1/models/decoder.py
src/phenotype_network_v1/models/losses.py
scripts/v1/03_train_conditioned_model.py
tests/v1/models/_m3_helpers.py        (test-only shared fixtures)
tests/v1/models/test_phenotype_conditioning.py
tests/v1/models/test_seed_injection.py
tests/v1/models/test_rwr_residual.py
```

## 3. Model specification coverage (Section 7)

* 7.1 Query encoder: attention pool over weighted HPO embeddings with masked
  absent terms and exposed normalised attention weights; weighted mean is the
  `weighted_mean_hpo_pool` ablation.
* 7.2 FiLM-conditioned two-layer basis R-GCN (8 bases, LayerNorm residual,
  GELU), per-layer `gamma(q)*m + beta(q)` conditioning.
* 7.2 Recurrent seed injection: `h_seed <- rho*seed_state + (1-rho)*h_seed` at
  every layer (default rho=0.2); disabling rho or seed injection are ablations.
* 7.3 Decoder: `score = MLP([h_g, q_d, h_g*q_d, log1p(rwr)])`; frozen V0 RWR
  is an optional residual column (`no_rwr_residual` ablation drops it).

Required ablations implemented as CLI flags:
`no_query_conditioning`, `condition_decoder_only`,
`no_recurrent_seed_injection`, `weighted_mean_hpo_pool`, `no_rwr_residual`.

## 4. Acceptance commands and results

### 4.1 Full test suite

```bash
.venv/Scripts/python.exe -m pytest -q
```

Result: **144 passed** (V0 82 + V1 62). M3 tests verify that distinct
phenotype queries on the same graph produce distinct gene scores
(`test_distinct_queries_produce_distinct_gene_scores`), deterministic replay
is identical (`test_deterministic_replay`), seed injection changes backbone
output, rho=0 reproduces no injection, and RWR-residual decoding behaves as
specified.

### 4.2 CPU end-to-end smoke (development machine)

```bash
PYTHONPATH=src .venv/Scripts/python.exe scripts/v1/03_train_conditioned_model.py --fold 0 --smoke
```

Result: exit 0; ablation=full params=7,794,106 device=cpu; train loss
0.4818 -> 0.1516 and val loss 0.2109 -> 0.0509 over 2 epochs; checkpoint,
manifest, and summary written under `outputs/v1/conditioned/full/fold_0/`.

## 5. Server follow-up (full runs on GPU)

Train `full` plus all five ablations across folds 0-4 on the GPU server
(`--device cuda`) with the Section 7.4 default configuration; record parameter
counts, validation-selected epochs, and metric tables. Ablation flags are
ready; CPU smoke validated each path end to end.

## 6. Deviations / notes

* Basis coefficients are shared across the two layers in this implementation
  (documented in `conditioned_rgcn.py`).
* V0 RWR rankings cover the diseases present in `outputs/rankings/...`; any
  disease/gene without a frozen RWR record maps to an RWR feature of 0 and is
  logged for audit (no recomputation inside training).

## 7. Acceptance result

- [x] pytest 144 passed
- [x] Distinct phenotype queries produce distinct gene scores (same graph)
- [x] Deterministic replay verified
- [x] Ablations selectable; seed/rho/FiLM/RWR behaviours unit-tested
- [x] CPU end-to-end smoke run writes checkpoint + manifest
- [ ] Server full training + ablation matrix (gpu) - deferred to GPU server
- [x] Acceptance document written; milestone stops here
