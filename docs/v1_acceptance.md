# V1 Acceptance (V1-A, Milestones 0-8)

Date: 2026-09-08
Branch: `codex/v1`
Git commit: see the M8 commit (on top of `67fd09b`)

## 1. Milestones implemented in order

| Milestone | Result |
|---|---|
| M0 Server scaffold and reproducibility | `00_check_environment.py --device cpu` exit 0 |
| M1 Fold-specific learning dataset | 5/5 fold leakage reports PASS (10/10 checks) |
| M2 Reproducible deep baselines | GCN/R-GCN/HGT CPU-verified; upstream gate recorded (Speos/XGDAG unavailable locally, never claimed reproduced) |
| M3 Phenotype-conditioned primary model | distinct queries -> distinct scores; deterministic replay; ablations selectable |
| M4 Bias-aware PU training and tuning | validation-only access-logged search; degree-stratified bagging PU audit |
| M5 Pathway hypergraph modules | soft modules + rejection audit; V0 Leiden comparator untouched |
| M6 Learned path explanations | approved-relation bounded paths + counterfactual fidelity |
| M7 External validation and robustness | external enrichment + perturbation stability; no retraining |
| M8 Final evaluation and release | metrics + verify_v1 gate PASS |

## 2. Final commands (development machine)

```bash
pytest -q                     # 183 passed, 1 skipped (gpu marker, no CUDA)
pytest -q -m gpu              # 1 skipped (CUDA absent locally)
python scripts/v1/verify_v1.py --config configs/v1/default.yaml   # PASS (33 checks)
```

`verify_v1.py` covers required files, all present fold leakage reports,
serialization hashes, query instances, checkpoint presence (fold 0 on this
machine), metric completeness (fold 0 smoke), portability (no absolute
paths), and claim boundaries (V1-B not started; no causal claim; GTEx support
is not causal; upstream methods without a passed smoke test are not claimed
as reproduced).

## 3. CPU smoke results (pipeline validation only - NOT scientific results)

CPU smoke runs on this development machine used 2-epoch checkpoints and tiny
query subsets; their metrics must not be interpreted as model quality:

* M8 fold 0 smoke (3 test queries): V1 conditioned/ensemble MRR ~0.0001,
  V0 raw-RWR MRR ~0.334; bootstrap delta negative. This smoke model is far
  below the V0 comparator and below the scientific performance gate.
* Per section 13, a passing engineering pipeline does not imply a positive
  scientific result; failure to meet the scientific gate remains visible here
  and in the model card until full GPU training is evaluated.

## 4. Server follow-up required before any scientific or deployment claim

1. Full GPU training: deep baselines (M2), conditioned model full + ablations
   (M3), 5-member PU ensembles per fold (M4), module-regularised run (M5).
2. Multi-fold explanations (M6), external validation and robustness (M7).
3. `pytest -q -m gpu` on CUDA; `00_check_environment.py --device cuda`.
4. Upstream unchanged smoke runs for Speos/XGDAG in pinned environments;
   review Speos license; XGDAG comparison only, no copying.
5. Re-run `verify_v1.py`; report final metrics vs V0 and deep baselines with
   statistical comparisons; keep negative results visible.

## 5. Claim boundaries

* No causal claim from graph scores, modules, paths, or attention.
* GTEx expression is tissue support, not causal proof.
* V1-B clinical extension is reserved (schema only) and not started: no
  patient cohort exists; SUA/LDL-C/hs-CRP/IVUS values are never generated.
* External methods without a passed unchanged smoke test are unavailable and
  are not presented as reproduced.
