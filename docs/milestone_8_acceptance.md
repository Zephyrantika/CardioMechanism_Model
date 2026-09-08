# Milestone 8 Acceptance

Phenotype-seeded random walk with restart (RWR) uses each fold's row-stochastic sparse transition matrix and train-derived HPO information content. Query seeds are normalized frequency-times-IC weights on phenotype nodes only. Validation diseases select alpha from the preregistered grid `[0.1, 0.2, 0.3, 0.5, 0.7]`; test labels never enter propagation or tuning.

All five folds converged at tolerance `1e-10`. Selected alpha values were `[0.2, 0.1, 0.1, 0.2, 0.1]`, with test iteration counts `[95, 198, 202, 106, 191]`. The original 200-iteration ceiling was raised to 500 because fold 2 required 202 iterations at its validation-selected alpha; the algorithm and tolerance were unchanged. Maximum observed probability-mass error was `3.21e-14`, and no query required uniform seed fallback.

Complete rankings cover 451 disjoint test diseases and 16,606 candidate genes: 7,489,306 rows. Cross-fold verification confirmed exact test membership, continuous ranks, no duplicate disease-gene pairs, finite nonnegative scores, descending score order, and hashes matching the effective fold configuration and transition matrix.

Unit tests cover alpha-one behavior, convergence, mass conservation, determinism, nonnegativity, propagation direction, seed weighting and fallback, input validation, and complete deterministic ranking. Full test suite: 56 passed.
