# Phenotype Network V1 Implementation Plan

## 1. Purpose And Execution Contract

This document is the executable specification for Phenotype Network V1. Codex
must implement exactly one milestone per task, in order, and must run the full
test suite after every milestone.

V1 extends the accepted V0 baseline with phenotype-conditioned graph learning.
It must not overwrite, rename, or silently alter V0 data, scripts, outputs,
folds, metrics, or acceptance documents.

When starting a milestone, Codex must first read:

```text
AGENTS.md
V1_IMPLEMENTATION_PLAN.md
IMPLEMENTATION_PLAN.md
docs/v0_acceptance.md
docs/server_migration.md
```

The task prompt should be:

```text
Implement and accept V1 Milestone N exactly as specified in
V1_IMPLEMENTATION_PLAN.md. Do not start later milestones.
```

If a required input is absent, Codex must stop that milestone with an explicit
missing-input report. It must not synthesize biomedical labels, patient values,
or positive results.

## 2. Scientific Objective

Given a disease phenotype profile, learn a leakage-aware ranking of candidate
genes while retaining auditable phenotype, pathway, module, and path evidence.
Unlike V0, phenotype information must condition every message-passing layer,
not only initialize a random walk.

V1 has two separately gated scopes:

- **V1-A, public disease-level model:** executable with the current public V0
  resources and a GPU server. Completion requires Milestones 0-8.
- **V1-B, clinical extension:** executable only after a compliant patient-level
  cohort is supplied. It is not required for V1-A acceptance.

## 3. Non-Goals And Claim Boundaries

- Do not use a large language model as the predictor or source of gene labels.
- Do not treat every unlabelled gene as a confirmed negative.
- Do not use GWAS, GTEx, GO, BioGRID sensitivity results, or patient outcomes
  for hyperparameter selection unless a milestone explicitly assigns a
  training-safe role.
- Do not expose test or validation disease-gene labels, equivalent relations,
  parent-child duplicates, or held-out family labels during pretraining.
- Do not claim causality from graph scores, attention, embeddings, modules, or
  paths. Causal language requires separate genetic, longitudinal, or
  perturbational evidence.
- Do not add serum uric acid, LDL-C, hs-CRP, IVUS plaque burden, or minimum lumen
  area as learned inputs without patient-level values and V1-B approval.
- Do not replace the V0 evaluator. V1 must be evaluated on the frozen V0 folds
  and candidate universe for direct comparison.

## 4. Starting State

The accepted V0 commit is:

```text
4a4a508 Complete interpretable phenotype network V0
```

V0 provides five family-disjoint folds, 451 cardiovascular diseases, 16,606
candidate genes, and one graph per fold with 38,258 nodes:

```text
19,836 phenotype nodes
16,606 gene nodes
1,816 Reactome pathway nodes
```

The primary V0 comparator is raw RWR. Its frozen aggregate performance is:

```text
Recall@10 = 0.0903
MRR       = 0.0535
```

V1 code must use a new package and script namespace:

```text
src/phenotype_network_v1/
scripts/v1/
configs/v1/
tests/v1/
outputs/v1/
data/processed/v1/
data/graphs/v1/
```

Existing `src/phenotype_network_v0/` functions may be imported when their
contracts are stable. Do not copy them without a documented reason.

## 5. Engineering And Server Rules

- Use Python 3.11 and Linux as the authoritative V1 training platform.
- Add a separate optional dependency group named `deep`; do not add PyTorch to
  V0 core dependencies.
- Initial framework: PyTorch and PyTorch Geometric. Pin Linux/CUDA versions in
  `requirements-v1-lock-linux.txt` after environment validation.
- CPU-only unit tests must run without CUDA and use synthetic fixtures.
- GPU integration tests must be marked `gpu` and skipped when CUDA is absent.
- All training runs must accept `--seed`, `--device`, `--fold`, `--config`,
  `--resume`, and `--output-dir` where applicable.
- Use deterministic algorithms where supported. Record any nondeterministic CUDA
  operation in QC.
- Use automatic mixed precision only after an FP32 equivalence smoke test.
- Checkpoints must contain model state, optimizer state, epoch, configuration
  hash, data hashes, seed, Git commit, and framework/CUDA versions.
- Never commit checkpoints, raw data, patient data, or generated outputs.
- All filesystem paths must use `pathlib.Path` and support the existing storage
  environment variables.
- Structured logging is required; `print` is forbidden in application code.

Recommended server allocation for initial experiments:

```text
GPU: 1 x NVIDIA A100 40 GB or 80 GB
CPU RAM: 64 GB recommended
Disk: 100 GB working space
CPU: 16 cores recommended for preprocessing
```

## 6. Data And Leakage Contract

### 6.1 Frozen Comparison Dataset

Every final V1 model must rank the same 16,606 candidate genes for the same V0
test diseases. The V0 `train_diseases.txt`, `val_diseases.txt`, and
`test_diseases.txt` files are immutable inputs.

### 6.2 Optional Pretraining Corpus

V1 may construct a broader human HPO disease corpus to reduce label scarcity.
For each fold, remove before graph serialization:

1. all gene labels for validation and test diseases;
2. all gene labels for diseases in their MONDO family groups;
3. equivalent or cross-reference relations that directly recover those labels;
4. reified duplicates of held-out disease-gene relations;
5. external-validation relations assigned only to final validation.

The fold-specific pretraining graph must be hashed independently. A single
global pretrained checkpoint trained with all disease labels is forbidden.

### 6.3 Query Instance Schema

Write `data/processed/v1/fold_N/query_instances.parquet` with:

```text
fold
split
disease_id
disease_family_ids
hpo_ids
hpo_weights
positive_gene_ids
covered_positive_gene_ids
unresolved_positive_gene_ids
source_version
```

HPO weights must continue to use frequency multiplied by fold-specific training
IC. Positive genes must use canonical NCBI Gene IDs.

### 6.4 Positive-Unlabelled Policy

Unknown disease-gene pairs are unlabelled, not confirmed negative. The primary
training strategy is deterministic degree-stratified bagging PU:

- five ensemble members by default;
- each member samples unlabelled genes within graph-degree deciles;
- default sampled-unlabelled to positive ratio: 20:1;
- each member uses an explicit derived seed;
- predictions are averaged before ranking;
- sampled records retain `label_role=sampled_unlabelled` in audit output.

Uniform random-negative training is allowed only as a bias ablation.

### 6.5 External Validation Isolation

GWAS Catalog, GTEx, GO/GOA, and BioGRID remain external evidence or sensitivity
resources. They may not select epochs, hyperparameters, thresholds, or model
architecture. Their source versions and SHA256 hashes must be frozen.

## 7. Primary Model Specification

### 7.1 Query Phenotype Encoder

For disease `d`, construct a query vector from weighted HPO embeddings:

```text
q_d = AttentionPool({w_dh * e_h : h in HPO(d)})
```

The attention pool must mask absent HPO terms and expose normalized weights for
audit. A weighted mean is the required non-attention ablation.

### 7.2 Phenotype-Conditioned R-GCN

The primary backbone is a two-layer basis-decomposed R-GCN. For each layer,
relation-specific messages are conditioned by `q_d` using FiLM gates:

```text
m_v^(l) = sum_r sum_(u,v,r) norm(u,v,r) * W_r^(l) h_u^(l)
h_v^(l+1) = LayerNorm(
    h_v^(l) + activation(gamma_l(q_d) * m_v^(l) + beta_l(q_d))
)
```

Inject the phenotype seed state at every layer:

```text
h_d^(l+1) = rho * seed_d + (1 - rho) * h_d^(l+1)
```

This recurrent seed injection is the operational definition of phenotype as a
persistent constraint. A model that uses the query only in the final decoder
does not satisfy the V1 objective.

### 7.3 RWR Residual

The gene decoder must combine learned and V0 signals:

```text
score(d,g) = MLP([h_g, q_d, h_g * q_d, log1p(rwr_score(d,g))])
```

An ablation without the RWR feature is mandatory. V0 RWR values are frozen
inputs and are never recomputed inside gradient training.

### 7.4 Default Configuration

```yaml
hidden_dimension: 128
layers: 2
relation_bases: 8
dropout: 0.20
seed_injection_rho: 0.20
activation: gelu
normalization: layer_norm
optimizer: adamw
learning_rate: 0.001
weight_decay: 0.00001
maximum_epochs: 300
early_stopping_patience: 30
disease_batch_size: 4
gradient_clip_norm: 1.0
ensemble_members: 5
sampled_unlabelled_ratio: 20
```

Permitted validation-only search space:

```yaml
hidden_dimension: [64, 128, 256]
layers: [2, 3]
dropout: [0.1, 0.2, 0.4]
seed_injection_rho: [0.1, 0.2, 0.3]
learning_rate: [0.0003, 0.001]
```

Use a fixed successive-halving budget. Test metrics must not be loaded by the
tuning process.

### 7.5 Required Deep Comparators

- homogeneous GCN over collapsed relations;
- R-GCN without phenotype FiLM or recurrent seed injection;
- HGT with matched parameter budget;
- Speos using its pinned upstream implementation and an adaptation to the
  frozen V0 folds;
- XGDAG using its pinned upstream implementation and an adaptation to the
  frozen V0 folds.

All direct comparators must rank the same candidate genes and obey the same
split and leakage contract. Report parameter count, training compute, upstream
commit, and every adaptation for each comparator. Rao et al. and Het2Gene are
not mandatory comparators because their currently available artifacts do not
meet the reproducibility gate in Section 12.

## 8. Module And Explanation Specification

### 8.1 Pathway Hypergraph Constraint

Reactome pathways act as hyperedges over genes. Add losses for:

- pathway membership reconstruction;
- within-module embedding compactness;
- perturbation consistency;
- anti-hub regularization;
- soft assignment entropy, preventing collapse into one module.

Module assignments must be query-specific. V0 Leiden modules remain a required
post-hoc comparator.

### 8.2 Path Reasoning

Replace hand-scored path ranking with a BioPathNet/NBFNet-style relation-aware
path reasoner restricted to evidence templates approved in V0. The reasoner
must output relation sequences, source databases, edge weights, and path scores.

Attention weights alone are not accepted as explanations. Explanation fidelity
must be tested with:

- critical-edge deletion;
- top-path sufficiency;
- top-path comprehensiveness;
- seed perturbation stability;
- comparison against randomized paths of the same length.

## 9. Evaluation Protocol

### 9.1 Primary Ranking Metrics

```text
Recall@1/5/10/20
MRR
Average Precision
AUPRC
NDCG@10
candidate coverage
```

Uncovered known genes must remain in an audit table and may not disappear from
metric denominators without an explicit covered-only companion result.

### 9.2 Stability And Bias Metrics

- Top-20 and Top-100 Jaccard under HPO deletion, PPI deletion, edge-weight
  noise, and network-source switching;
- Spearman correlation between score and graph degree;
- performance by gene-degree decile and disease-family size;
- variance across at least three training seeds for final candidates;
- PU ensemble disagreement and rank uncertainty.

### 9.3 Explanation Metrics

- resolved Top-20 target coverage;
- critical-edge deletion score change;
- path sufficiency and comprehensiveness;
- path stability under perturbation;
- Reactome/GO support rate using frozen external evidence.

### 9.4 Statistical Analysis

Use paired disease-level differences and cluster bootstrap by disease family
with 1,000 replicates. Pre-register raw RWR and the strongest deep comparator
as the two primary comparisons. Correct secondary comparisons with Holm's
method.

### 9.5 Acceptance Gates

Technical acceptance requires all tests, leakage checks, deterministic replay,
artifact hashes, and output schemas to pass.

Scientific success requires:

1. V1 primary model exceeds raw RWR on Recall@10 or MRR in at least 3/5 folds;
2. the paired family-bootstrap 95% interval excludes zero for at least one of
   the two co-primary metrics;
3. the improvement is not explained by candidate coverage;
4. score-degree correlation is no worse than the uncalibrated V0 model;
5. conclusions retain direction under STRING 700/900 sensitivity;
6. at least one explanation-fidelity measure improves over V0 paths.

If technical acceptance passes but the scientific gate fails, publish the run
as a negative result and do not tune on the test folds.

## 10. Milestones

### V1 Milestone 0: Server Scaffold And Reproducibility

### Objective

Create the isolated V1 package, configuration, dependency, test, logging, and
checkpoint foundations without implementing a graph model.

### Create

```text
configs/v1/data.yaml
configs/v1/model.yaml
configs/v1/training.yaml
configs/v1/default.yaml
src/phenotype_network_v1/__init__.py
src/phenotype_network_v1/environment.py
src/phenotype_network_v1/checkpoint.py
scripts/v1/00_check_environment.py
tests/v1/test_environment.py
tests/v1/test_checkpoint.py
requirements-v1.in
requirements-v1-lock-linux.txt
docs/v1_server_migration.md
```

Update `pyproject.toml` only to include the new package, `deep` dependency
group, and `gpu` pytest marker. Update `.gitignore` for checkpoints and V1
outputs.

Create or switch to `codex/v1` before editing unless the user names another V1
branch. Inspect the server driver first, then select an officially compatible
PyTorch/CUDA/PyG combination and freeze the resolved Linux environment. Do not
copy a Windows lock file to the server.

### Acceptance

```bash
python scripts/v1/00_check_environment.py --device cpu
pytest -q
```

On the server, also run `--device cuda`. The report must record Python,
PyTorch, PyG, CUDA, cuDNN, GPU model, deterministic settings, and free memory.

Stop after writing `docs/v1_milestone_0_acceptance.md`.

### V1 Milestone 1: Fold-Specific Learning Dataset

### Objective

Convert frozen V0 folds and graphs into auditable query instances and
fold-specific PyG graph artifacts without label leakage.

### Create

```text
src/phenotype_network_v1/data/query_dataset.py
src/phenotype_network_v1/data/pyg_graph.py
src/phenotype_network_v1/data/leakage.py
scripts/v1/01_build_learning_data.py
tests/v1/data/test_query_dataset.py
tests/v1/data/test_v1_leakage.py
data/processed/v1/fold_N/query_instances.parquet
data/graphs/v1/fold_N/heterodata.pt
data/graphs/v1/fold_N/graph_manifest.json
data/graphs/v1/fold_N/leakage_report.json
```

### Required Tests

- canonical IDs and V0 node-index roundtrip;
- disease and family separation;
- no validation/test labels or equivalent duplicate relations;
- identical candidate universe across models;
- train-only IC and phenotype-gene supervision;
- deterministic graph serialization hash;
- unlabelled genes never marked confirmed negative.

### Acceptance

All five leakage reports pass. Graph counts reconcile with V0 except for
explicitly documented training-safe additions. Stop after
`docs/v1_milestone_1_acceptance.md`.

### V1 Milestone 2: Reproducible Deep Baselines

### Objective

Implement GCN, unconditional R-GCN, and parameter-matched HGT comparators, then
verify and adapt the retained Speos and XGDAG upstream baselines. All five use
the same training, early-stopping, candidate, and evaluation contracts after
adaptation.

### Create

```text
src/phenotype_network_v1/models/gcn.py
src/phenotype_network_v1/models/rgcn.py
src/phenotype_network_v1/models/hgt.py
src/phenotype_network_v1/baselines/upstream.py
src/phenotype_network_v1/training/trainer.py
src/phenotype_network_v1/training/sampling.py
configs/v1/upstream_methods.yaml
scripts/v1/02_train_baselines.py
scripts/v1/02_verify_upstream.py
tests/v1/models/test_baselines.py
tests/v1/training/test_early_stopping.py
tests/v1/baselines/test_upstream_manifest.py
docs/v1_literature_reproducibility.md
```

Before adaptation, record each repository URL, immutable commit SHA, license,
runtime, data source, and expected command in `upstream_methods.yaml`. Run one
unchanged upstream smoke test and preserve its command, log, and output hash.
XGDAG has no repository license as of the audit date: execute the pinned source
for comparison only and do not copy or redistribute it. Any failed upstream
smoke test blocks that external baseline; do not silently replace it.

After adaptation, run one CPU synthetic test and one server smoke run for every
model. Document input mappings, code changes, unavailable upstream data, and
metric deviations. Validation selects epochs; test labels must remain unread
until the frozen checkpoint is loaded by the evaluator.

Stop after `docs/v1_milestone_2_acceptance.md`.

### V1 Milestone 3: Phenotype-Conditioned Primary Model

### Objective

Implement the query encoder, FiLM-conditioned R-GCN, recurrent phenotype seed
injection, and RWR residual decoder defined in Section 7.

### Create

```text
src/phenotype_network_v1/models/phenotype_encoder.py
src/phenotype_network_v1/models/conditioned_rgcn.py
src/phenotype_network_v1/models/decoder.py
src/phenotype_network_v1/models/losses.py
scripts/v1/03_train_conditioned_model.py
tests/v1/models/test_phenotype_conditioning.py
tests/v1/models/test_seed_injection.py
tests/v1/models/test_rwr_residual.py
```

### Required Ablations

```text
no_query_conditioning
condition_decoder_only
no_recurrent_seed_injection
weighted_mean_hpo_pool
no_rwr_residual
```

The model must produce different gene scores for distinct phenotype queries on
the same graph and identical scores for deterministic replay.

Stop after `docs/v1_milestone_3_acceptance.md`.

### V1 Milestone 4: Bias-Aware PU Training And Tuning

### Objective

Add deterministic degree-stratified bagging PU, ensemble uncertainty, and the
fixed validation-only hyperparameter search.

### Create

```text
src/phenotype_network_v1/training/pu.py
src/phenotype_network_v1/training/search.py
src/phenotype_network_v1/evaluation/bias.py
scripts/v1/04_tune_and_train_ensemble.py
tests/v1/training/test_pu_sampling.py
tests/v1/training/test_search_isolation.py
tests/v1/evaluation/test_degree_bias.py
```

### Acceptance

- five ensemble members complete for every fold;
- every sampled-unlabelled record is auditable;
- validation-only tuning is proven by access logs;
- repeated seeds reproduce samples and model selection;
- uniform random-negative ablation is clearly labelled non-primary.

Stop after `docs/v1_milestone_4_acceptance.md`.

### V1 Milestone 5: Pathway Hypergraph Modules

### Objective

Add query-specific Reactome hypergraph regularization and extract stable soft
mechanism modules without replacing the V0 Leiden comparator.

### Create

```text
src/phenotype_network_v1/models/pathway_hypergraph.py
src/phenotype_network_v1/models/module_losses.py
src/phenotype_network_v1/evaluation/module_stability.py
scripts/v1/05_train_modules.py
tests/v1/models/test_hypergraph.py
tests/v1/evaluation/test_module_stability.py
```

Reject collapsed, single-hub-dominated, unstable, or unsupported modules into
an audit table. Module labels must come from frozen Reactome/GO evidence.

Stop after `docs/v1_milestone_5_acceptance.md`.

### V1 Milestone 6: Learned Path Explanations

### Objective

Implement relation-aware path reasoning and counterfactual fidelity evaluation.

### Create

```text
src/phenotype_network_v1/models/path_reasoner.py
src/phenotype_network_v1/evaluation/explanation_fidelity.py
scripts/v1/06_build_explanations.py
tests/v1/models/test_path_reasoner.py
tests/v1/evaluation/test_explanation_fidelity.py
```

Restrict paths to approved biomedical relations and maximum lengths. Preserve
unresolved targets. Do not use held-out labels to search or score paths.

Stop after `docs/v1_milestone_6_acceptance.md`.

### V1 Milestone 7: External Validation And Robustness

### Objective

Evaluate frozen V1 checkpoints against GWAS, GTEx, GO/GOA, BioGRID, STRING 900,
and all preregistered perturbations without retraining.

### Create

```text
src/phenotype_network_v1/evaluation/external.py
src/phenotype_network_v1/evaluation/robustness.py
scripts/v1/07_validate_and_perturb.py
tests/v1/evaluation/test_external_isolation.py
tests/v1/evaluation/test_robustness.py
```

GTEx expression is tissue support, not causal proof. GWAS support must record
the locus-to-gene mapping rule. External sources may not change candidate
membership or model weights.

Stop after `docs/v1_milestone_7_acceptance.md`.

### V1 Milestone 8: Final Evaluation And Release

### Objective

Run the frozen evaluator, statistical comparisons, acceptance gates, packaging,
and migration verification.

### Create

```text
src/phenotype_network_v1/evaluation/metrics.py
src/phenotype_network_v1/evaluation/statistics.py
scripts/v1/08_evaluate_v1.py
scripts/v1/verify_v1.py
docs/v1_acceptance.md
docs/v1_model_card.md
docs/v1_reproducibility.md
```

### Final Commands

```bash
pytest -q
pytest -q -m gpu
python scripts/v1/verify_v1.py --config configs/v1/default.yaml
```

`verify_v1.py` must check required files, all fold leakage reports, checkpoint
hashes, model/data configuration hashes, metric completeness, primary
comparisons, explanation fidelity, external isolation, portability, and claim
boundaries.

Tag the release only after the worktree is clean and acceptance is complete.

## 11. V1-B Clinical Extension: Hard Gate

V1-A must ship with a reserved clinical-extension interface, but the
interface is schema-only until a compliant cohort is supplied. An empty or
absent clinical input must be a valid state and must not change V1-A model
outputs, graph hashes, folds, or metrics.

### 11.1 Reserved Interface Contract

The future adapter should expose the following typed boundary:

```python
load_clinical_cohort(source: Path, schema: Path) -> ClinicalCohort
validate_clinical_cohort(cohort: ClinicalCohort) -> ClinicalAudit
split_clinical_cohort(cohort: ClinicalCohort, seed: int) -> ClinicalSplits
```

The planned long-format input schema is:

```text
patient_id, visit_id, site_id, visit_datetime,
disease_id, feature_id, value, unit, missing_reason, source_record_id
```

`feature_id` must use the existing registry identifiers:
`CLIN:SUA`, `CLIN:LDL_C`, `CLIN:HS_CRP`,
`CLIN:IVUS_PLAQUE_BURDEN`, and `CLIN:IVUS_MIN_LUMEN_AREA`. The adapter must
preserve original values, units, source records, and unresolved mappings in an
audit table. It must not silently convert units or map a clinical measurement
to an HPO term without a recorded mapping rule.

The reserved implementation boundary is:

```text
src/phenotype_network_v1/clinical/schema.py
src/phenotype_network_v1/clinical/adapter.py
src/phenotype_network_v1/clinical/split.py
configs/v1/clinical.yaml
```

These files may be added later at V1-B/B0. V1-A may include only the schema,
template, and validation stubs; it must not contain patient values, raw IVUS
images, identifiers, or clinical outcomes.

The disease-level graph and V1-A checkpoints remain frozen inputs. V1-B may
add a patient phenotype/measurement encoder and an independent validation
head, but it may not alter the disease-level candidate universe or retrain on
test-site or future-time records. Patient, site, and time-aware splits are
mandatory, and imputation, normalization, feature selection, and batch
correction must be fitted on the training split only.

Do not implement V1-B until all of the following are supplied:

```text
ethics/IRB or permitted-use documentation
de-identification statement
cohort data dictionary
patient and visit identifiers
site and acquisition dates
serum uric acid values and units
LDL-C values and units
hs-CRP values and units
IVUS plaque burden and minimum lumen area definitions
missingness report
outcome definitions
train/validation/test splitting policy
```

V1-B must use patient-, site-, and time-aware splitting. Imputation,
normalization, feature selection, and batch correction must be fitted on the
training split only. Repeated visits from one patient may not cross splits.

Recommended V1-B sequence:

1. B0 governance and schema validation;
2. B1 clinical feature preprocessing and missingness audit;
3. B2 patient phenotype encoder with disease-level graph frozen;
4. B3 optional multi-omics or single-cell context encoder;
5. B4 external-site or temporal validation.

If the cohort is unavailable, retain the 15-row V0 clinical bridge as context
only and report V1-B as not started.

## 12. Reproducible Literature Baseline

Audit snapshot: 2026-08-05. Retain a method only when the paper is published in
a leading general or computational-biology journal and a paper-matched public
artifact provides source code, runnable entry points, usable data or an explicit
data acquisition path, and an environment description. A public repository is
not sufficient by itself.

### 12.1 Retained Methods

| Class | Method | Evidence and V1 obligation |
| --- | --- | --- |
| `required_reproduction` | Speos, Nature Communications, `10.1038/s41467-023-42975-z` | `github.com/fratajcz/speos`; explicit license, Conda/Docker environments, data downloader, training and benchmark commands. Reproduce upstream, then run as a direct comparator. |
| `required_reproduction` | XGDAG, Bioinformatics, `10.1093/bioinformatics/btad482` | `github.com/GiDeCarlo/XGDAG`; Conda environments, data, pretrained models, training and ranking entry points. Reproduce upstream and compare, but do not copy or redistribute because the repository has no license. |
| `method_component_reuse` | BioPathNet, Nature Biomedical Engineering, `10.1038/s41551-025-01598-z` | `github.com/emyyue/BioPathNet`; MIT license, pinned requirements, mock data, train/predict/visualize commands. Reproduce its mock run before adapting path reasoning; it is not a direct disease-gene ranking comparator. |
| `method_component_reuse` | Network Enhancement, Nature Communications, `10.1038/s41467-018-05469-x` | `github.com/wangboyunze/Network_Enhancement`; GPL-3.0, MATLAB examples and bundled data. Use only as a network-denoising ablation; a Python port requires numerical parity tests against the upstream output. |
| `method_component_reuse` | CoLiPE bias-aware evaluation, PNAS, `10.1073/pnas.2416646122` | `github.com/serhan-yilmaz/CoLiPE`; GPL-3.0, bundled inputs and MATLAB demo entry points. Reuse its degree-bias evaluation design, not its code, unless MATLAB reproduction is recorded. |
| `method_component_reuse` | Network-propagation benchmark, PLOS Computational Biology, `10.1371/journal.pcbi.1007276` | `github.com/b2slab/genedise`; MIT, data, R/Matlab workflows and reproducibility notes. Use its evaluation protocol as a benchmark cross-check, not as a deep comparator. |

### 12.2 Not Required For Reproduction

| Class | Method | Reason |
| --- | --- | --- |
| `discussion_only` | Rao et al., `10.1186/s12920-018-0372-8` | Historically relevant, but the paper's official `gcas.tar.gz` artifact is no longer available at its stated URL. |
| `discussion_only` | End-to-end interpretable disease-gene prediction, `10.1093/bib/bbad118` | No verifiable paper-matched public repository with license, environment, data, and runnable entry points was found. |
| `exclude` | Phenotype-driven hypergraph prioritization, `10.1038/s41598-025-04428-z` | The code-availability statement provides only a contact address, not public code. |
| `discussion_only` | Het2Gene, `10.1016/j.compbiomed.2026.111543` | Public artifacts contain inference code and data but no pinned environment or documented end-to-end training reproduction; it also does not improve the retained journal tier. |
| `exclude` | MORE, `10.1093/bib/bbae658` | The repository lacks a license and a complete reproducible environment, and the task is biomedical classification rather than disease-gene ranking. |

### 12.3 Reproducibility Gate

Milestone 2 must freeze the repository URL and commit SHA, record the license and
data terms, run the documented upstream smoke command before modification, and
hash its log and outputs. Adapted runs must document all deviations. A method
that fails this gate is reported as unavailable and cannot be presented as a
reproduced result. `discussion_only` and `exclude` papers create no coding or
comparison obligation.

## 13. Completion Definition

V1-A is complete only when Milestones 0-8 are implemented in order, all tests
and leakage checks pass, the server run is reproducible from pinned
dependencies, the final model is compared against V0 and deep baselines, and
the result is reported without causal or clinical overclaiming.

A passing engineering pipeline does not imply a positive scientific result.
Failure to meet the scientific performance gate must remain visible in
`docs/v1_acceptance.md` and the model card.
