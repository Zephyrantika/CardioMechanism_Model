# Milestone 9 Acceptance

Each fold calibrates test RWR scores against 1,000 deterministic bootstrap null queries drawn only from its family-disjoint training diseases. The 25-profile matched pool minimizes four preregistered components: phenotype count, global HPO depth, train-IC quartiles, and top-level clinical-system Jaccard distance. Temperature `0.5` was selected after a five-fold audit: it increased the minimum number of distinct null profiles from 5 to 24 while retaining substantially closer matches than temperatures 1.0 or 2.0.

HPO depth is measured from `HP:0000001`. Phenotypic abnormalities use organ-system children of `HP:0000118`; clinical modifiers, inheritance, and history terms retain their global top-level categories. Broad root annotations remain explicitly marked `UNRESOLVED` rather than being forced into an organ system. Null RWR runs in batches of 64 for local/server portability without changing the propagation result.

Formal outputs cover 451 disjoint test diseases, 16,606 genes, 7,489,306 calibrated ranking rows, and 451,000 null-match audit rows. Every query has 1,000 draws and at least 24 distinct training profiles. All batches converged at `1e-10` with probability-mass error below `1e-10`. Original RWR scores are preserved exactly. Rankings contain null mean/std, Z-score, midrank empirical percentile, PPI and graph degree, raw and corrected ranks, and row-level zero-variance flags.

Zero variance affects 119,515 rows (1.596%; 265 genes per query) and uses a documented Z-score fallback of zero. Median raw score-PPI degree Spearman correlation is 0.8691; the empirical-percentile-corrected median is 0.0025. Full test suite: 62 passed.
