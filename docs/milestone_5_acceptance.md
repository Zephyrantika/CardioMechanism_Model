# Milestone 5 Acceptance

The leaf-level benchmark contains 451 diseases assigned to 216 deterministic MONDO family anchors at registered depth 4. Five folds contain 91/90/90/90/90 diseases. Experiment `k` uses fold `k` for test, fold `(k + 1) mod 5` for validation, and the remaining three folds for training.

Each fold has a train-only HPO IC table and train-only HPO–Gene edges with contributing disease IDs retained for audit. The candidate universe is the fixed union of STRING 700 and Reactome genes: 16,606 genes with one SHA256 hash across all folds. Test positive-gene candidate coverage ranges from 94.68% to 98.91%; missing positives remain listed as unevaluable.

All checks pass in every `leakage_report.json`: disease sets and family sets are disjoint, no validation/test disease contributes to HPO–Gene edges, no parent-child disease pair crosses folds, no direct answer edge exists, and candidate universes are identical. Full test suite: 42 passed.