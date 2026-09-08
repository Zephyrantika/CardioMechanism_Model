# Milestone 6 Acceptance

Each of the five fold graphs contains 38,258 nodes: 19,836 HPO terms, 16,606 candidate genes, and 1,816 Reactome pathways. Transition matrices contain 640,040–645,334 nonzero values. All weights are nonnegative and every row sums to one; the maximum observed normalization error is `6.66e-16`.

Relations are independently normalized before source-type budgets are applied. Missing outgoing relation types trigger proportional budget redistribution, and isolated rows would receive self-loops; no fully isolated node remained in the real graph. HPO and pathway hierarchies are bidirectional structural relations. Supervised HPO–Gene relations are train-only and fold-specific.

Cross-fold hashes prove that HPO hierarchy, STRING, Reactome gene-pathway, and pathway hierarchy matrices are identical. Only HPO→Gene, Gene→HPO, and the resulting transition matrix differ by fold. Full test suite: 46 passed.