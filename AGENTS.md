# Project instructions

## Project goal

Build a reproducible, leakage-aware phenotype-driven heterogeneous
network pipeline for disease gene prioritization.

The first version is an interpretable baseline. Do not introduce deep
learning, graph neural networks, large language models, or GPU
dependencies unless a later task explicitly requests them.

## Engineering rules

- Use Python 3.11.
- Use `src` layout.
- Use `pathlib.Path` for paths.
- Use type annotations for public functions.
- Use structured logging instead of print statements.
- Keep raw, interim, processed, graph, and output data separated.
- Never overwrite files under `data/raw`.
- Use Parquet for processed tables unless a task specifies otherwise.
- Use SciPy sparse matrices for full graph propagation.
- NetworkX may only be used for small explanation subgraphs.
- Every data transformation must record input counts, output counts,
  removed records, and unresolved identifiers.
- All randomized operations must accept a seed.
- Do not silently discard malformed or unmapped records. Write them to
  an audit table.

## Biomedical data rules

- Canonical phenotype identifier: HPO ID.
- Canonical disease identifier: MONDO ID when a reliable mapping exists.
- Canonical gene identifier: NCBI Gene ID.
- Canonical pathway identifier: Reactome stable ID.
- Preserve original identifiers and source databases.
- Never treat every unlabelled gene as a confirmed negative.
- Never mix external validation data into the training graph.
- Never include test disease–gene labels or equivalent duplicate
  relations in the background graph.
- Do not claim causal biological conclusions from network predictions.

## Testing rules

Run all tests after every task:

`pytest -q`

Add tests for:

- identifier mapping;
- obsolete-term replacement;
- duplicate-edge removal;
- graph matrix dimensions;
- transition-matrix normalization;
- train/test disease separation;
- label leakage;
- RWR convergence and determinism.

Use only small synthetic fixtures in automated tests. Do not require
large external downloads to run unit tests.

## Task discipline

- Implement only the requested milestone.
- Do not create speculative modules for later milestones.
- Do not replace explicit algorithms in `IMPLEMENTATION_PLAN.md`
  without documenting the reason.
- When requirements are ambiguous, choose the simplest deterministic
  implementation and document the assumption.