# V1 Model Card (V1-A, phenotype-conditioned disease-gene ranking)

## Summary
Phenotype-conditioned graph model (query HPO attention encoder + FiLM
basis R-GCN with recurrent seed injection + RWR-residual gene decoder) for
ranking candidate genes for cardiovascular diseases on the frozen V0
five-fold benchmark. V1-B (clinical) is not started.

## Intended use
Research on leakage-aware disease-gene prioritization; public disease-level
inputs only. Not for clinical decisions, diagnosis, or treatment.

## Training data
Fold-specific learning datasets derived from frozen V0 folds (M1). Labels are
literature/curated disease-gene associations; unlabelled genes are NOT
confirmed negatives (PU policy, M4).

## Evaluation
Frozen V0 folds; Recall@k and MRR vs V0 raw RWR and deep baselines (M2),
external support (M7), path fidelity (M6), module audits (M5).

## Known limitations (current)
* CPU smoke checkpoints (2 epochs, tiny subsets) are far below the V0
  comparator and the scientific gate (see `docs/v1_acceptance.md` section 3).
* Full GPU training and scientific evaluation are pending on the GPU server.
* GTEx evidence is tissue support, not causal.
* External methods Speos/XGDAG are not yet reproduced (no local environment;
  XGDAG has no license and is never copied).

## Fairness / ethics
No patient data; V1-B requires IRB/permitted-use documentation and a
de-identification statement before any clinical cohort work.

## Caveat
This model card does not claim clinical or causal utility. A passing
engineering pipeline is not a positive scientific result.
