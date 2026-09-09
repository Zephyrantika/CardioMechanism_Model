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

---

## 6. GPU full-training results (2026-09-09, local RTX 4060 8GB)

All five folds trained with the M3 default recipe (AdamW 1e-3, BCE on covered
positive genes, per-query SGD, early stopping patience 30 on val loss,
best-epoch checkpoints ~epoch 2-3, full 30+ epochs before stop):

| Fold | V1 MRR | V1 R@10 | V0 RWR MRR | V0 RWR R@10 | V1/V0 MRR |
|---|---|---|---|---|---|
| 0 | 0.0196 | 0.0337 | 0.0400 | 0.0787 | 0.49 |
| 1 | 0.0096 | 0.0133 | 0.0332 | 0.0244 | 0.29 |
| 2 | 0.0318 | 0.0696 | 0.0905 | 0.1253 | 0.35 |
| 3 | 0.0296 | 0.0455 | 0.0522 | 0.0795 | 0.57 |
| 4 | 0.0179 | 0.0556 | 0.0593 | 0.1111 | 0.30 |
| mean | 0.0217 | 0.0435 | 0.0550 | 0.0838 | 0.39 |

### Honest scientific statement (section 13)
The current M3 recipe does NOT meet the scientific performance gate: on every
fold V1 is well below the frozen V0 raw-RWR comparator (mean MRR 0.0217 vs
0.0550). Training quickly overfits (validation loss rises after epoch 2-3),
which indicates the recipe - not the architecture alone - needs improvement.

### Recorded recipe hypotheses for the next iteration (not yet run)
* degree-stratified PU negative sampling during training (M4 machinery exists),
* learning-rate schedule / lower LR with longer warmup,
* larger hidden dimension / FiLM capacity regularisation (dropout, weight decay),
* query-encoder pretraining or stronger phenotype dropout augmentation,
* validation-based hyperparameter search already implemented (M4) should select
  these knobs instead of defaults.
A negative engineering result stays visible here until a recipe beats V0 RWR.

### 6.1 Recipe iteration results (fold 0, 2026-09-09)

| Recipe | best epoch | V1 MRR | V1 R@10 | note |
|---|---|---|---|---|
| default (BCE all genes, lr 1e-3) | 2 | 0.0196 | 0.0337 | best of the tried recipes so far |
| A: PU (deg-strat., ratio 20) lr 1e-3 | 1 | 0.0054 | 0.0112 | worse |
| B: PU lr 3e-4 | 3 | 0.0089 | 0.0337 | worse |
| C: PU lr 1e-4 (patience 40) | 2 | 0.0045 | 0.0000 | worse |
| V0 raw RWR | - | 0.0400 | 0.0787 | target |

All recipes still overfit within 1-3 epochs (validation loss rises immediately),
and none reaches the V0 comparator on fold 0. Recorded hypothesis: overfitting
is not mainly a missing-negative-sampling problem; per-query memorisation is
strong at this task scale. Architecture-level changes (self-supervised graph
pretraining / link prediction, stronger regularisation, phenotype
augmentation) are the next research stage, outside the acceptance framework.
### 6.2 Second iteration (fold 0): regularisation / capacity variants

| Recipe | best epoch | V1 MRR | V1 R@10 |
|---|---|---|---|
| D: lr 3e-4, dropout 0.4, wd 1e-2 | 2 | 0.0148 | 0.0337 |
| E: hidden 256, dropout 0.3, lr 3e-4 | 3 | 0.0091 | 0.0112 |
| default (best so far) | 2 | 0.0196 | 0.0337 |
| V0 raw RWR | - | 0.0400 | 0.0787 |

Neither stronger regularisation (D) nor larger capacity (E) reaches V0 on
fold 0. Scientific gate: NOT MET with the recipes tried. The next stage is
architecture-level research (self-supervised graph pretraining / link
prediction objectives), which is outside the milestone acceptance framework
and is documented here as an open problem.
