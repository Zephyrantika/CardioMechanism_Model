# V1 Milestone 0 Acceptance: Server Scaffold And Reproducibility

Date: 2026-09-08
Branch: `codex/v1`
Git commit: `e67314c8700c99cb25b71c36bb5615575331fe7e`
Acceptance scope: CPU scaffold verified on the development machine; the CUDA
device check is specified for the GPU server (see section 5).

## 1. Objective

Create the isolated V1 package, configuration, dependency, test, logging, and
checkpoint foundations without implementing a graph model. V0 files are
untouched; V1 lives in isolated namespaces.

## 2. Created files

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

Updated (only as specified): `pyproject.toml` (adds `phenotype_network_v1`
package, `deep` optional dependency group, `gpu` pytest marker),
`.gitignore` (V1 outputs, checkpoints, `*.pt`/`*.pth`).

## 3. Acceptance commands and results

### 3.1 Environment check (CPU)

```bash
PYTHONPATH=src .venv/Scripts/python.exe scripts/v1/00_check_environment.py --device cpu
```

Result: exit code **0**, device check `OK`. Report written to
`outputs/v1/environment_report_cpu.json`.

### 3.2 Full test suite

```bash
.venv/Scripts/python.exe -m pytest -q
```

Result: **103 passed** in 3.66 s (V0 suite 82 passed + 21 new V1 tests).

## 4. Environment record

| Component | Value |
|---|---|
| Python | 3.13.0 (CPython) |
| Platform | Windows-11-10.0.26200-SP0 (development machine) |
| PyTorch | not installed (deep group optional on CPU profile) |
| PyG | not installed |
| CUDA | not available |
| cuDNN | not available |
| GPU model | none |
| Deterministic settings | not configured (no PyTorch) |
| Free memory | n/a (no GPU) |

The structured report also records the Git commit
(`e67314c8700c99cb25b71c36bb5615575331fe7e`) and package version (0.1.0).

## 5. GPU server follow-up (required before V1-A training milestones)

The server environment check (`--device cuda`) and `gpu`-marked tests must be
run on the authoritative Linux/NVIDIA server following
`docs/v1_server_migration.md` section 3:

```bash
.venv/bin/python scripts/v1/00_check_environment.py --device cuda
pytest -q
```

The curated Linux lock (`requirements-v1-lock-linux.txt`) pins PyTorch
2.11.0 + cu128, PyG 2.8.0, pyg-lib 0.7.0 for Python 3.11; it must be replaced
by a full resolved `pip freeze` after server validation.

## 6. Configuration and data hashes

Config files (SHA-256, as committed):

```text
1fae18a25fa89a64153d56e0d6cc78884a3c050d4144d39a413e5e74dd1d3c61  configs/v1/data.yaml
9a06a61fdbc928e2809581895cbe24c6d753599c62d042bd71f9c5b9585561f8  configs/v1/default.yaml
a6a5262d4ca8fc28061c6574fc57f21271bd236b2742622cc3f01409ec709e8e  configs/v1/model.yaml
28f2e8f2ede40c62f1c38ad88dace681f7b3b77c681371073173e07b49d88579  configs/v1/training.yaml
1ff923a03f5970317e673656629c6d7fd9f66693833104b57810b6570cbf2eca  requirements-v1.in
2bcbde2e59daf3546c269f68e517ee261bb502bb6597bff26db806d4c45905ad  requirements-v1-lock-linux.txt
```

Frozen data contract (read from `configs/v1/data.yaml`): 5 folds, 451
diseases, 16,606 candidate genes, 38,258 graph nodes per fold. No V0 data or
folds were modified; no raw, patient, checkpoint, or generated output is
committed.

## 7. Acceptance result

- [x] `scripts/v1/00_check_environment.py --device cpu` exits 0
- [x] `pytest -q` passes (103 passed)
- [ ] `--device cuda` on GPU server (deferred; no CUDA hardware locally)
- [x] Acceptance document written; milestone stops here (no later milestone started)
