# V1 Milestone 5 Acceptance: Pathway Hypergraph Modules

Date: 2026-09-08
Branch: `codex/v1`
Commit: (M5 commit, on top of `ddc7ba9`)
Scope: query-specific Reactome hypergraph regularisation, soft module
extraction, stability, and audit; V0 Leiden comparator untouched.

## 1. Created files

```text
src/phenotype_network_v1/models/pathway_hypergraph.py
src/phenotype_network_v1/models/module_losses.py
src/phenotype_network_v1/evaluation/module_stability.py
scripts/v1/05_train_modules.py
tests/v1/models/test_hypergraph.py
tests/v1/evaluation/test_module_stability.py
```

## 2. Design notes

* Modules are query-specific pathway activations: activation = sum of the
  query's sigmoid gene scores over frozen Reactome primary-graph membership
  (`data/processed/reactome_gene_pathway.parquet`). Membership is never
  derived from test labels.
* Rejection audit: `collapsed`, `oversized`, `single_hub_dominated`,
  `unstable`, `unsupported` reasons are written to `module_audit.parquet`.
* The V0 Leiden comparator is retained unchanged (not replaced).

## 3. Commands and results

### 3.1 Full test suite

```bash
.venv/Scripts/python.exe -m pytest -q
```

Result: **167 passed** (V0 82 + V1 85).

### 3.2 CPU end-to-end smoke

```bash
PYTHONPATH=src .venv/Scripts/python.exe scripts/v1/05_train_modules.py --fold 0 --smoke
```

Result: exit 0; loaded the M3 conditioned checkpoint; 3 test queries -> 24
modules, all retained; `module_qc.json` and `modules.parquet` written under
`outputs/v1/modules/fold_0/`.

## 4. Server follow-up

Run module extraction on all folds with fully trained conditioned checkpoints
and enable `--train-with-module-loss` for the module-regularised run; report
retained/rejected distributions and stability across folds.

## 5. Acceptance result

- [x] pytest 167 passed
- [x] Query-specific pathway activations from frozen Reactome evidence
- [x] Soft modules with stability and audit (rejection reasons)
- [x] V0 Leiden comparator untouched
- [x] CPU smoke end to end
- [ ] Full multi-fold module training/extraction (gpu/server) - deferred
- [x] Acceptance document written; milestone stops here
