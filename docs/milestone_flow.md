# Milestone Flow

This diagram is the execution contract for V0. Solid arrows are required pipeline dependencies; dashed arrows are validation or future-extension boundaries.

```mermaid
flowchart TD
    M0["M0 Engineering initialization"]
    M1["M1 Full HPO ontology"]
    M2["M2 Cardiovascular benchmark<br/>and clinical feature registry"]
    M3["M3 STRING PPI<br/>optional BioGRID sensitivity"]
    M4["M4 Reactome pathways<br/>optional GO annotations"]
    M5["M5 MONDO family-disjoint folds<br/>and train-only IC"]
    M6["M6 Sparse heterogeneous graph"]
    M7["M7 Resnik / BMA baseline"]
    M8["M8 Phenotype-driven RWR"]
    M9["M9 Network bias correction"]
    M10["M10 Evaluation and ablation"]
    M11["M11 Immunometabolic modules"]
    M12["M12 Evidence-constrained paths"]
    M13["M13 Atherosclerosis, MI, and HF cases"]

    GWAS["Frozen GWAS Catalog"]
    GTEX["Frozen cardiovascular GTEx"]
    GO["Frozen GO annotations"]
    CLIN["SUA / LDL-C / hs-CRP<br/>IVUS-derived metrics"]
    FUTURE["Future de-identified clinical cohort"]

    M0 --> M1 --> M2
    M0 --> M3
    M0 --> M4
    M2 --> M5
    M3 --> M5
    M4 --> M5
    M5 --> M6
    M5 --> M7
    M6 --> M8 --> M9 --> M10
    M7 --> M10
    GWAS -. external validation only .-> M10
    GTEX -. external validation only .-> M10
    GO -. enrichment and annotation .-> M10
    M10 --> M11 --> M12 --> M13
    CLIN -. registered clinical bridge .-> M13
    M13 -. independent future validation .-> FUTURE
```

## Stage Gates

- **M2 gate:** canonical identifiers, audit tables, cardiovascular inclusion rules, and clinical-feature mappings are complete.
- **M5 gate:** all MONDO family groups are disjoint and every leakage check passes.
- **M10 gate:** test folds remain untouched by training and tuning; external resources remain frozen and independent.
- **M13 gate:** reports distinguish predictions, database support, clinical associations, and causal claims.

## Scope Boundary

V0 trains only on public disease-level annotations and propagates across HPO, Gene, and Reactome Pathway nodes. Clinical numeric variables do not become graph nodes. The future clinical cohort is a separate validation extension, not an additional training input for V0.