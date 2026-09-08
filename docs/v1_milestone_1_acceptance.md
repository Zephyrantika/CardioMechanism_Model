# V1 Milestone 1 Acceptance: Fold-Specific Learning Dataset

Date: 2026-09-08
Branch: `codex/v1`
Git commit (code): `4fb5c89` (M1 implementation, on top of `62a44ba`)
Scope: CPU-only conversion of frozen V0 folds into V1 query instances and
PyG graph artifacts. No graph model is implemented.

## 1. Objective

Convert frozen V0 folds and graphs into auditable query instances and
fold-specific PyG graph artifacts without label leakage.

## 2. Created files

```text
src/phenotype_network_v1/data/__init__.py
src/phenotype_network_v1/data/query_dataset.py
src/phenotype_network_v1/data/pyg_graph.py
src/phenotype_network_v1/data/leakage.py
scripts/v1/01_build_learning_data.py
tests/v1/data/test_query_dataset.py
tests/v1/data/test_v1_leakage.py
```

Generated artifacts (never committed; gitignored):

```text
data/processed/v1/fold_N/query_instances.parquet
data/graphs/v1/fold_N/heterodata.pt
data/graphs/v1/fold_N/graph_manifest.json
data/graphs/v1/fold_N/leakage_report.json
```

## 3. Acceptance commands and results

### 3.1 Learning-data build (all five folds)

```bash
PYTHONPATH=src .venv/Scripts/python.exe scripts/v1/01_build_learning_data.py --fold all
```

Result: exit code **0**; all five fold leakage reports **PASS (10/10 checks)**:

| Fold | Query rows | Leakage | Nodes | Edges (reconciles V0) |
|---|---|---|---|---|
| 0 | 451 | PASS 10/10 | 39,065 | 745,918 |
| 1 | 451 | PASS 10/10 | 39,065 | 742,292 |
| 2 | 451 | PASS 10/10 | 39,065 | 743,996 |
| 3 | 451 | PASS 10/10 | 39,065 | 746,096 |
| 4 | 451 | PASS 10/10 | 39,065 | 747,656 |

Edge counts match the frozen V0 graph QC (`transition_nnz` per fold) exactly;
V1 adds no training-safe edges in this milestone.

### 3.2 Full test suite

```bash
.venv/Scripts/python.exe -m pytest -q
```

Result: **115 passed** (V0 82 + V1 33) in 7.14 s.

## 4. Query instance statistics (identical per fold)

* 2255 query rows total (451 diseases x 5 folds; train ~270 + validation 90 + test 90-91)
* Positive gene labels: 686 per fold (canonical NCBI Gene IDs)
  * covered by frozen candidate universe: 674
  * unresolved (outside candidate universe): 12 - never used as negatives
* HPO weights = `frequency_weight * fold-specific training IC`
* No confirmed-negative labels anywhere; unlabelled policy is recorded in
  the audit (`unlabelled_never_confirmed_negative: true`)

## 5. Reconciliation with the V1 plan "Starting State"

V1_IMPLEMENTATION_PLAN.md section 4 quotes representative V0 counts
(19,836 phenotype / 16,606 gene / 1,816 pathway = 38,258 nodes; 16,606
candidate genes). The frozen V0 outputs delivered and accepted on this
machine use slightly different counts. V1 uses the frozen V0 data as the
source of truth; the difference is recorded, not silently corrected:

```text
observed per fold: 19,894 phenotype + 16,992 gene + 2,179 pathway = 39,065 nodes
frozen candidate universe: 16,992 genes per fold
```

All fold partitions, disease counts (451), split files, and leakage endorsements
come from the accepted V0 folds and are unchanged.

## 6. Required-test coverage (V1 plan Milestone 1)

| Requirement | Covered by |
|---|---|
| Canonical IDs and V0 node-index roundtrip | `test_v0_node_index_roundtrip` |
| Disease and family separation | `disease_sets_disjoint` / `disease_families_disjoint` checks + tests |
| No validation/test labels or equivalent duplicate relations | `v0_fold_leakage_endorsed` + graph reconcile + negative tests |
| Identical candidate universe across models | `positive_genes_covered_or_unresolved` + universe hash audit |
| Train-only IC and phenotype-gene supervision | `hpo_ids_in_fold_training_ic` / `hpo_weights_nonnegative_finite` |
| Deterministic graph serialization hash | `test_heterodata_deterministic_serialization` + manifest `serialization_sha256` |
| Unlabelled genes never marked confirmed negative | `no_confirmed_negative_labels` check + `test_positive_never_negative` |

## 7. Environment and hashes

* Python 3.13.0 (Windows development machine); PyTorch 2.11.0+cpu and PyG
  2.8.0 installed locally for CPU graph construction only.
* Per-fold `graph_manifest.json` records source-input SHA-256 (node_map,
  graph_qc, all `A_*.npz`), serialization SHA-256 of `heterodata.pt`, and the
  Git commit at build time.
* Leakage reports record candidate-universe and training-IC SHA-256 per fold.

## 8. Acceptance result

- [x] 5/5 fold leakage reports PASS (10/10 checks each)
- [x] Graph counts reconcile with frozen V0 (edges identical; nodes per V0)
- [x] `pytest -q`: 115 passed
- [x] Acceptance document written; milestone stops here (no later milestone started)
