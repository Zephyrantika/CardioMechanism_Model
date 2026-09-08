# V1 Literature Reproducibility (Milestone 2)

Audit snapshot: 2026-08-05 (V1_IMPLEMENTATION_PLAN.md section 12). This
document records the upstream baseline and component-reuse obligations, the
reproduction gate, and the status on the current machine. Machine-readable
state lives in `configs/v1/upstream_methods.yaml`.

## 1. Reproduction gate (section 12.3)

Before adaptation each retained method must freeze its repository URL, commit
SHA, license and data terms, run one unchanged upstream smoke test, and hash
its log and outputs. A method that fails this gate is reported as unavailable
and **cannot be presented as a reproduced result**. `discussion_only` and
`exclude` papers create no coding or comparison obligation.

## 2. Deep baselines implemented in-repo (no upstream copy)

The required deep comparators are implemented from scratch in this repository
(no copied third-party source):

| Model | Module | Implementation notes |
| --- | --- | --- |
| GCN | `src/phenotype_network_v1/models/gcn.py` | Homogeneous GCNConv baseline over the unified V1 graph |
| R-GCN | `src/phenotype_network_v1/models/rgcn.py` | Relation-specific linear messages with self loop (RGCN semantics) |
| HGT | `src/phenotype_network_v1/models/hgt.py` | Relation-conditioned multi-head transformer attention (PyG `TransformerConv`, edge_dim = relation embedding). HGT-style deviation note: relation identities condition attention through edge attributes rather than full per-metapath type-specific linear/attention projection of the original HGT. Recorded as an adaptation deviation. |

All three share the training, early-stopping, candidate, and evaluation
contracts and are parameter-matched (default hidden 64, 2 layers; heads=4 for
HGT). Parameter counts are logged per run.

## 3. Retained external methods (status table)

Status legend: `not_run_local_no_environment` = unchanged smoke test not yet
executed on this machine (no Conda/MATLAB/R environment); such methods are
**unavailable** and not presented as reproduced.

| Method | Class | License | Runtime | Commit SHA (frozen) | Status |
| --- | --- | --- | --- | --- | --- |
| Speos (`10.1038/s41467-023-42975-z`) | required_reproduction | unknown_pending_review (GitHub NOASSERTION; in-repo LICENSE must be reviewed) | conda/docker | `9618b361` | not run (no environment) |
| XGDAG (`10.1093/bioinformatics/btad482`) | required_reproduction | none (do not copy/redistribute; execute pinned source for comparison only) | conda/python | `1b801a5d` | not run (no environment) |
| BioPathNet (`10.1038/s41551-025-01598-z`) | method_component_reuse | MIT | python | `b051a900` | not run (no environment) |
| Network_Enhancement (`10.1038/s41467-018-05469-x`) | method_component_reuse | GPL-3.0 | MATLAB | `86622848` | not run (no environment) |
| CoLiPE (`10.1073/pnas.2416646122`) | method_component_reuse | GPL-3.0 | MATLAB | `ee74162e` | not run (no environment); reuse design only, not code |
| genedise (`10.1371/journal.pcbi.1007276`) | method_component_reuse | MIT | R/MATLAB | `10074e24` | not run (no environment); benchmark cross-check only |

## 4. What must happen on the GPU server before any "reproduced" claim

1. Install each retained method's pinned environment (Conda/Docker).
2. Run the documented unchanged smoke command; preserve command, log, output
   hash, and data acquisition path.
3. For XGDAG execute the pinned commit for comparison only; never copy or
   redistribute its code (no license).
4. For Speos review the in-repo LICENSE before any adaptation; record the
   license identifier.
5. Only after an unchanged smoke test passes may a method be listed as
   `passed_unchanged` in `configs/v1/upstream_methods.yaml`.

Until then the V1 report must describe these methods as unavailable external
comparators, not as reproduced results.
