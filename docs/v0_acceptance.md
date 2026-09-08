# V0 Acceptance

Phenotype Network V0 is complete as an interpretable, CPU-only baseline. Milestones 0-13 have been implemented and accepted without deep learning, graph neural networks, large language models, GPU dependencies, or patient-level modeling.

The frozen benchmark contains 451 cardiovascular diseases evaluated in five family-disjoint folds. The primary raw RWR model exceeds the best semantic baseline on Recall@10 or MRR in all five folds. Its aggregate Recall@10 is 0.0903 and MRR is 0.0535. Test disease-gene labels remain evaluation-only, and frozen BioGRID, GWAS Catalog, GTEx, GO, and GOA resources do not enter training or hyperparameter selection.

Module analysis completes 250 deterministic perturbation RWR runs and retains 2,661 non-hub-dominated modules across all diseases. Explanation analysis covers all 9,020 Top-20 targets: 8,684 have constrained evidence paths and 336 are explicitly unresolved. All 8,684 critical-edge deletion runs converge. Paths and module annotations are evidence summaries, not causal claims.

The three preregistered case studies are correctly reported as unevaluable because the frozen public benchmark lacks exact HPO and known-gene inputs. No labels are manually added. Their 15 clinical-feature bridge rows are isolated from training and reserved for contextual or future independent validation use.

Final verification passes all 15 cross-milestone checks. The automated suite passes with `82 passed`. Dependencies are frozen in `requirements-lock.txt`, raw-data roots support environment overrides, source and configuration files contain no local absolute paths, and `docs/server_migration.md` documents server transfer and verification.
