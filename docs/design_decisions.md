# Design Decisions

These decisions constrain later milestones but do not introduce their implementations.

## V0 Scope and Clinical Boundary

- V0 is a public-data, disease-level cardiovascular immunometabolism baseline. Its primary cases are coronary atherosclerosis, myocardial infarction, and heart failure.
- The propagation graph remains HPO, NCBI Gene, and Reactome Pathway only. Patient-level measurements, metabolites, immune-cell nodes, and raw IVUS images are outside V0.
- SUA, LDL-C, hs-CRP, plaque burden, and minimum lumen area are registered separately. Reliable HPO or LOINC mappings are retained; otherwise a documented local canonical ID is used without forcing an HPO mapping.

## Data Sources and Identifiers

- Reactome processing will also require `ReactomePathways.txt` for pathway names and species. The hierarchy file is not a metadata source.
- `genes_to_phenotype.txt` is optional for Milestone 2 and may only support QC or sensitivity analyses.
- MONDO mappings require explicit ontology xrefs or unambiguous one-to-one mappings. Unresolved records retain identifiers such as `OMIM:xxxxxx` with `mapping_method = unresolved_to_mondo`; fuzzy name matching is prohibited.
- STRING mappings prefer explicit Ensembl/NCBI evidence. Ambiguous one-to-many mappings are audited. Isoforms are collapsed at gene level, duplicate PPI evidence channels use their maximum, and original protein IDs and evidence are retained.

## Evaluation Population

The candidate universe for each fold is the union of cleaned STRING and Reactome gene nodes. Test labels never expand this universe. Missing positives are reported as unevaluable, candidate coverage is reported, and ranking metrics are also reported over evaluable positives.

Only leaf-level benchmark diseases are supervised; broader benchmark labels that are ancestors of another eligible disease are audited and excluded. Five fixed disease folds are used. A deterministic MONDO family anchor at registered depth 4 defines indivisible groups; canonical duplicates are merged and any remaining parent-child pair must stay in one fold. For experiment `k`, test is fold `k`, validation is fold `(k + 1) mod 5`, and the remaining three folds are training data. Stratification uses disease family and binned phenotype and positive-gene counts. If five valid family groups cannot be formed, the pipeline fails rather than splitting or duplicating families.

## Phenotypes and Leakage

Primary HPO information content is computed per fold from training-disease annotations in Milestone 5. Global IC is sensitivity-only. Frequency parsing will support percentages, ratios, ranges (using the midpoint), HPO frequency terms, and missing values. Unknown frequency weighs 0.5; unparseable non-empty values are audited with their raw value and parsing method.

Training graphs exclude test disease-gene labels, reverse or duplicate answer relations, HPO-gene edges derived from test labels, and test-label-derived features. Test genes may remain in ordinary STRING, Reactome, and label-free HPO structures.

## Graph Construction and QC

Relation budgets are combined per source-node row using type masks. Missing relation budgets are redistributed across available outgoing relation types; fully isolated nodes receive self-loops. Final transition-matrix rows must sum to approximately one.

Unmapped warning and failure thresholds are data-source-specific, configurable, and reported by the milestone that introduces each source.

## Frozen External Validation

GWAS Catalog, cardiovascular-relevant GTEx tissues, and GO annotations are version-frozen before final evaluation. They do not enter training, hyperparameter selection, or the candidate universe. GTEx expression is tissue-context support, not standalone causal evidence.

## Future Direction

A later clinical extension may use a de-identified cohort and derived IVUS numeric measurements for independent association or external validation. Patient outcomes and continuous measurements must remain separate from the disease-level training graph. V1 may then evaluate phenotype-conditioned, learnable long-range propagation; APPNP is one possible foundation, not a V0 dependency.
## Milestone 10 Primary Ranking Decision

Raw RWR remains the preregistered primary predictor. Empirical-percentile calibration removes degree association but substantially reduces held-out gene recovery, so calibrated rank is retained only as a bias-sensitivity analysis and its Z-score may be used as one normalized module-scoring component. Milestone 11 therefore selects Top-200 candidates by raw RWR; this avoids post hoc replacement of the primary predictor while preserving an explicit degree-bias diagnostic.
