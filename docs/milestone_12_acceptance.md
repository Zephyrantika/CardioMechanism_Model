# Milestone 12 Acceptance

Milestone 12 builds constrained evidence paths for the Top-20 raw RWR genes of every held-out disease. Paths use only the preregistered HPO-HPO, HPO-gene, gene-gene, and gene-pathway relations from each fold's training graph. Test disease-gene labels are never exposed to path construction.

The completed run covers 451 diseases and 9,020 target genes. It produces 25,141 paths for 8,684 targets; the remaining 336 targets are retained in `unresolved_explanation_targets.parquet` with the explicit reason `no_allowed_evidence_path`. Every resolved target has at most three simple paths, and every path matches an allowed template with no more than four edges.

For each resolved target, the highest-ranked path is checked by deleting its most costly edge and rerunning sparse RWR. All 8,684 deletion runs converged. A path is marked as the main explanation only when the target score has a positive relative contribution above the configured threshold. This yields 7,547 main explanations. Non-primary paths do not inherit critical-edge results.

Acceptance checks pass: target coverage is complete, all paths are simple, path counts and lengths are bounded, all deletion runs converge, unresolved targets are audited, causal interpretation is explicitly disabled, and no test labels are used. These paths are traceable evidence summaries, not causal biological claims.
