# V1 Milestone 4 Acceptance: Bias-Aware PU Training And Tuning

Date: 2026-09-08
Branch: `codex/v1`
Commit: (M4 commit, on top of `b3ff898`)
Scope: deterministic degree-stratified bagging PU, validation-only fixed-budget
search with access log, degree-bias evaluation. Full five-member-per-fold
training runs on the GPU server.

## 1. Created files

```text
src/phenotype_network_v1/training/pu.py
src/phenotype_network_v1/training/search.py
src/phenotype_network_v1/evaluation/bias.py
scripts/v1/04_tune_and_train_ensemble.py
tests/v1/training/test_pu_sampling.py
tests/v1/training/test_search_isolation.py
tests/v1/evaluation/test_degree_bias.py
```

## 2. Acceptance coverage

| Requirement | Evidence |
|---|---|
| Five ensemble members per fold | `04_tune_and_train_ensemble.py --members 5` (default); CPU smoke ran 2 members end to end |
| Sampled-unlabelled records auditable | `label_role=sampled_unlabelled` in `pu_audit.parquet` (438 rows in smoke) |
| Validation-only tuning proven by access logs | `AccessLog` forbids `test`-tagged access (tests pass); smoke log: `accesses=['validation_query_instances']` |
| Repeated seeds reproduce samples and selection | unit tests: identical seed/member reproduces identical samples; same-seed search returns identical best config |
| Uniform random-negative ablation clearly non-primary | `--sampling uniform` sets `primary=false` in `ensemble_summary.json` |

## 3. Commands and results

### 3.1 Full test suite

```bash
.venv/Scripts/python.exe -m pytest -q
```

Result: **159 passed** (V0 82 + V1 77).

### 3.2 CPU end-to-end smoke

```bash
PYTHONPATH=src .venv/Scripts/python.exe scripts/v1/04_tune_and_train_ensemble.py --fold 0 --smoke
```

Result: exit 0. Search (access-logged) -> 2 ensemble members trained -> audit
rows 438 -> summary written under
`outputs/v1/ensemble/degree_stratified/fold_0/`.

## 4. Server follow-up

Train five members per fold for all folds 0-4 (`--members 5`, full epochs,
`--device cuda`); keep the uniform-negative ablation run clearly labelled
non-primary; freeze the resulting ensemble checkpoints for M8 evaluation.

## 5. Acceptance result

- [x] pytest 159 passed
- [x] Deterministic degree-stratified bagging PU + audit roles
- [x] Validation-only tuning with access-log proof (test never read)
- [x] Seed reproducibility of samples and model selection
- [x] Uniform-negative ablation labelled non-primary
- [x] CPU smoke end to end (2 members)
- [ ] Full 5-member x 5-fold training (gpu) - deferred to GPU server
- [x] Acceptance document written; milestone stops here
