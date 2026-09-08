# V1 Milestone 6 Acceptance: Learned Path Explanations

Date: 2026-09-08
Branch: `codex/v1`
Commit: (M6 commit, on top of `8c92fd7`)

## 1. Created files

```text
src/phenotype_network_v1/models/path_reasoner.py
src/phenotype_network_v1/evaluation/explanation_fidelity.py
scripts/v1/06_build_explanations.py
tests/v1/models/test_path_reasoner.py
tests/v1/evaluation/test_explanation_fidelity.py
```

## 2. Behaviour

* Bounded, cycle-free path search over approved biomedical relations only
  (7 V0 relations), maximum length enforced, structure-only (no labels used).
* Counterfactual fidelity: each path edge is removed one at a time and the
  gene score drop is measured; unresolved targets are preserved and reported
  in `unresolved_targets.parquet` (never used as negatives).

## 3. Commands and results

`pytest -q` -> **175 passed**.
CPU smoke: `06_build_explanations.py --fold 0 --smoke` exit 0; 2 test queries,
5 paths, explanation_qc.json written.

## 4. Acceptance result

- [x] pytest 175 passed
- [x] Approved-relation, length-limited path reasoning
- [x] Counterfactual edge-deletion fidelity
- [x] Unresolved targets preserved
- [x] CPU smoke end to end
- [ ] Full multi-fold explanation run (server) - deferred
