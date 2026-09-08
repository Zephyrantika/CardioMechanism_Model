# V1 Reproducibility (Milestone 8)

## Code
Branch `codex/v1`; commits from `e67314c` (v1 handoff integration) through the
M8 commit; clean-worktree tag applied after acceptance.

## Dependencies
* V0 CPU: `requirements-lock.txt` (installed in `.venv`, Python 3.13 on the
  development machine).
* V1 deep: `requirements-v1.in` / `requirements-v1-lock-linux.txt` (Linux,
  Python 3.11, PyTorch 2.11.0+cu128, PyG 2.8.0, pyg-lib 0.7.0; curated - to
  be replaced by a full resolved lock after server validation).
* `deep` group in `pyproject.toml`; CPU-only unit tests never require CUDA.

## Data
Frozen V0 folds and graphs are read-only inputs (never regenerated). V1
generated artifacts (`data/processed/v1`, `data/graphs/v1`, `outputs/v1`) are
gitignored and hashed per fold in `graph_manifest.json` (serialization SHA-256)
and `leakage_report.json` (candidate universe / training IC SHA-256).

## Determinism
Every randomised run accepts `--seed`; member seeds are derived explicitly
(`derive_seed`); deterministic sampling is tested; deterministic graph
serialization is hashed; replay tests pass.

## Config and data hashes
Per-fold manifests and leakage reports record config/data hashes; checkpoints
carry M0 manifest schema (epoch, seed, config hash, data hashes, git commit,
framework versions).

## GPU server checklist
See `docs/v1_server_migration.md` and `docs/v1_acceptance.md` section 4.
Record `nvidia-smi`, `00_check_environment.py --device cuda` output, full
`pip freeze`, and all training/eval logs; update the lock and manifests.

## Known deferred items (transparent)
* GPU full training/evaluation and `pytest -m gpu` on CUDA.
* Upstream unchanged smoke runs for Speos/XGDAG (env/license-gated).
* V1-B clinical: not started (no compliant cohort).
