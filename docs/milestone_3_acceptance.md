# Milestone 3 Acceptance

## Result

STRING v12.0 was downloaded from the official STRING endpoint and validated to gzip EOF. Protein-to-gene mappings use only numeric `Ensembl_HGNC_entrez_id` aliases; ambiguous proteins are excluded and audited.

| Threshold | Gene edges | Gene nodes |
|---:|---:|---:|
| 400 | 913,092 | 19,111 |
| 700 | 232,869 | 15,952 |
| 900 | 99,583 | 12,180 |

The pre-registered primary network is STRING 700. It has 125 connected components; its largest component contains 15,642 nodes (98.06%). It covers 402 of 417 genes in the leaf-level Milestone 2 benchmark (96.40%). The top-degree list is recorded in `outputs/qc/ppi_qc.json` for later bias analysis.

## Audit

- 19,196 proteins have a unique NCBI Gene mapping.
- One protein has an ambiguous mapping and is excluded.
- 366 proteins referenced by retained-threshold edges lack a unique mapping and are audited.
- Eight protein edges collapse to gene self-loops and are removed.
- Duplicate undirected gene edges aggregate each evidence channel by maximum; raw source files, example protein pairs, pair counts, source, and version remain traceable.

## Verification

- `pytest -q`: 31 passed.
- No self-loops, duplicate undirected edges, invalid weights, or threshold nesting violations.
- Input/output count reconciliation passed.
- Peak private memory during the real run was approximately 1.8 GB; `--chunk-size` is configurable for server migration.