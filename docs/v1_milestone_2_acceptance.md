# V1 Milestone 2 Acceptance: Reproducible Deep Baselines

Date: 2026-09-08
Branch: `codex/v1`
Commit: `(M2 commit, on top of fa22ce6)` - recorded in the final commit
Scope: deep comparators implemented and CPU-verified; upstream gate recorded;
server training runs deferred to the GPU server.

## 1. Objective

Implement GCN, unconditional R-GCN, and parameter-matched HGT comparators and
record the reproduction status of the retained Speos/XGDAG/component-reuse
methods. All models share the same training, early-stopping, candidate, and
evaluation contracts after adaptation.

## 2. Created files

```text
src/phenotype_network_v1/models/__init__.py
src/phenotype_network_v1/models/gcn.py
src/phenotype_network_v1/models/rgcn.py
src/phenotype_network_v1/models/hgt.py
src/phenotype_network_v1/training/__init__.py
src/phenotype_network_v1/training/trainer.py
src/phenotype_network_v1/training/sampling.py
src/phenotype_network_v1/baselines/__init__.py
src/phenotype_network_v1/baselines/upstream.py
src/phenotype_network_v1/evaluation/__init__.py
configs/v1/upstream_methods.yaml
scripts/v1/02_train_baselines.py
scripts/v1/02_verify_upstream.py
tests/v1/models/test_baselines.py
tests/v1/training/test_early_stopping.py
tests/v1/baselines/test_upstream_manifest.py
docs/v1_literature_reproducibility.md
```

## 3. Acceptance commands and results

### 3.1 Full test suite

```bash
.venv/Scripts/python.exe -m pytest -q
```

Result: **133 passed** (V0 82 + V1 51). New M2 tests cover each baseline
(score shape, deterministic replay, distinct queries, tiny overfit), early
stopping, checkpoint manifest round trip, and the upstream manifest gate.

### 3.2 CPU synthetic test per model

Covered by `tests/v1/models/test_baselines.py`: every model scores all genes,
is deterministic for identical seed queries, and can overfit a single
synthetic query (all three pass).

### 3.3 CPU end-to-end smoke (development machine)

```bash
PYTHONPATH=src .venv/Scripts/python.exe scripts/v1/02_train_baselines.py --model gcn --fold 0 --smoke
```

Result: exit 0; model=gcn fold=0 params=2,508,609 device=cpu; train loss
0.7501 -> 0.6974 and val loss 0.7179 -> 0.6665 over 2 epochs; checkpoint and
manifest written under `outputs/v1/baselines/gcn/fold_0/`.

### 3.4 Upstream manifest verification

```bash
PYTHONPATH=src .venv/Scripts/python.exe scripts/v1/02_verify_upstream.py
```

Result: exit 0; reproduced=0; required-reproduction methods Speos and XGDAG
reported unavailable (unchanged smoke test not runnable locally: no
Conda/MATLAB/R environments). Their repository URLs, frozen commit SHAs,
licenses, and expected commands are recorded in
`configs/v1/upstream_methods.yaml` per the section 12.3 gate.

## 4. Server follow-up (required before any reproduced/deployed claim)

* Run full training (gcn/rgcn/hgt x 5 folds) on the GPU server with
  `--device cuda`, record parameter counts and validation-selected epochs.
* Run the unchanged upstream smoke for Speos and XGDAG in their pinned
  environments; update `smoke_status` to `passed_unchanged` only afterwards.
* Review Speos's in-repo LICENSE before adaptation (GitHub reports
  NOASSERTION); XGDAG (no license) is executed for comparison only, never
  copied or redistributed.

## 5. Deviations recorded

* HGT comparator is implemented as relation-conditioned multi-head attention
  on the unified graph (PyG `TransformerConv`, edge_dim = relation embedding)
  rather than the full original HGT per-metapath projections. Noted in
  `docs/v1_literature_reproducibility.md` section 2.
* Validation selects epochs; test labels are not read by any training code
  (checkpoint loaded only by the evaluator in later milestones).

## 6. Acceptance result

- [x] pytest 133 passed
- [x] CPU synthetic test for every model (gcn/rgcn/hgt)
- [x] CPU end-to-end smoke run writes checkpoint + manifest
- [x] Upstream manifest frozen and verified; unavailable methods not claimed as reproduced
- [ ] Server smoke runs (gpu) - deferred to GPU server
- [x] Acceptance document written; milestone stops here
