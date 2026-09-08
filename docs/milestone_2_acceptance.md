# Milestone 2 Acceptance

## Result

Milestone 2 passed on the HPO 2026-06-23 and MONDO 2026-07-06 releases. The benchmark contains 451 leaf-level cardiovascular diseases, 12,310 positive phenotype annotations, 18 negative phenotype annotations, and 688 disease-gene associations. Thirty-nine broader ancestor labels were excluded to prevent parent-child leakage. All input rows reconcile to an output, scope exclusion, malformed-record audit, duplicate, or threshold exclusion.

## Core Cases

The pre-registered cases `MONDO:0021661` (coronary atherosclerosis), `MONDO:0005068` (myocardial infarction), and `MONDO:0005252` (heart failure) have no joint disease-level HPO and gene labels in the selected HPO release. They remain registered for later public external-validation case reports and are not manually injected into training.

## Manual Review

| MONDO ID | Disease | Review |
|---|---|---|
| `MONDO:0012011` | coronary artery disease, autosomal dominant, 1 | MEF2A (`4205`); annotations include myocardial infarction and premature coronary atherosclerosis. |
| `MONDO:0012586` | coronary artery disease, autosomal dominant 2 | LRP6 (`4040`); annotations include myocardial infarction, gout, hypertension, and hypertriglyceridemia. |
| `MONDO:0700335` | familial isolated dilated cardiomyopathy | 52 canonical genes; annotations include dilated cardiomyopathy and congestive heart failure. |
| `MONDO:0008647` | hypertrophic cardiomyopathy 1 | MYH6, MYH7, MYLK2, and CAV3; annotations include septal hypertrophy and arrhythmia. |
| `MONDO:0979867` | recessive cerebral arteriopathy with subcortical infarcts | NOTCH3 (`4854`); 156 source annotations trace to `OMIM:621295`. |

Three disease-HPO pairs have conflicting positive and negative source annotations. Both polarities are retained and recorded in `data/interim/conflicting_phenotype_annotations.tsv`; negative annotations remain excluded from V0 propagation.

## Verification

- `pytest -q`: 27 passed.
- Canonical-ID, duplicate-edge, threshold, and QC-reconciliation checks passed.
- Source versions, file sizes, and SHA256 hashes are recorded in `outputs/qc/cardiovascular_benchmark_qc.json`.