# Milestone 11 Acceptance

Milestone 11 selects each disease's Top-200 genes from the frozen raw RWR primary ranking. Calibrated Z-scores contribute only a within-disease percentile component to module scoring; corrected rank is not used to replace the primary predictor.

For every fold, 50 deterministic perturbations rerun full sparse RWR: 15 delete 10% of each query's HPO seeds, 15 delete 10% of STRING PPI edges, 10 alternate STRING 700/900, and 10 apply 10% phenotype-frequency weight noise. All 250 runs converge under the frozen tolerance and maximum iteration settings. Each perturbed Top-200 subgraph is reclustered with weighted Leiden using an explicit seed.

Leiden produces 2,851 candidate modules. The preregistered size filter retains sizes 5–91. A weighted hub-dominance check rejects 190 modules to `rejected_modules.parquet`; no rejected module appears in final members or enrichment. The final output contains 2,661 modules across all 451 diseases and 83,331 memberships. Of these, 74,095 memberships meet the 0.60 consensus-frequency threshold.

Module score is `0.50 * mean calibrated-Z percentile + 0.25 * internal density + 0.25 * mean perturbation Jaccard`. Reactome v97 and GO/GOA frozen releases provide hypergeometric enrichment with within-module BH-FDR. The output contains 206,166 auditable enrichment rows; every retained module label is derived from an FDR-significant database term. Labels are evidence annotations, not causal mechanism claims.

Acceptance checks pass: all diseases covered, 250/250 convergence, deterministic seeds, module size 5–100, no retained single-hub module, explicit rejection audit, 0.60 consensus threshold, source-grounded enrichment, no test labels, and full tests (`81 passed`).