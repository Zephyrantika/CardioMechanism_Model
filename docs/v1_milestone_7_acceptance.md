# V1 Milestone 7 Acceptance: External Validation And Robustness

Date: 2026-09-08
Branch: `codex/v1`
Commit: (M7 commit, on top of `73b5155`)

## 1. Created files

```text
src/phenotype_network_v1/evaluation/external.py
src/phenotype_network_v1/evaluation/robustness.py
scripts/v1/07_validate_and_perturb.py
tests/v1/evaluation/test_external_isolation.py
tests/v1/evaluation/test_robustness.py
```

## 2. Behaviour

* Frozen conditioned checkpoints scored on test queries; top-k external
  support enrichment for GWAS / GTEx / GO / BioGRID / STRING-900.
* Mapping rules recorded: GWAS locus-to-gene rule from the frozen processed
  table; GTEx is tissue support only, not causal proof.
* Preregistered edge perturbations (explicit seeds) measure rank stability;
  no retraining; external sources never change candidate membership or model
  weights.

## 3. Commands and results

`pytest -q` -> **183 passed**.
CPU smoke: `07_validate_and_perturb.py --fold 0 --smoke` exit 0; 1 test query:
perturbation rank stability ~0.9995; GWAS support 36/50 in the top-50.

## 4. Acceptance result

- [x] pytest 183 passed
- [x] External enrichment computed; isolation preserved (no candidate/weight change)
- [x] GTEx non-causal statement and GWAS mapping rule recorded
- [x] Perturbation robustness with explicit seeds
- [x] CPU smoke end to end
- [ ] Full multi-fold external validation (server) - deferred
