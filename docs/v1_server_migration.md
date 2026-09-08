# V1 Server Migration And Reproducibility

Status: Milestone 0 scaffold (see `docs/v1_milestone_0_acceptance.md`).

This document describes how the V1 environment is reproduced on a GPU server
and how CPU-only development machines stay usable. It mirrors the intent of
`docs/server_migration.md` for the V0 baseline.

## 1. Environment split

V1 keeps two execution profiles:

| Profile | Machine | Purpose |
|---|---|---|
| CPU | development machine | Unit tests (`pytest -q`) and environment scaffold, no CUDA required |
| GPU | server (Linux, NVIDIA) | V1-A training runs, Milestones 2-8 |

The optional dependency group `deep` (see `pyproject.toml`) is only required
on the GPU profile. CPU-only unit tests must pass without it and are marked to
skip any GPU integration test (`@pytest.mark.gpu`) when CUDA is absent.

## 2. CPU-only environment (this checkout)

```bash
python -m venv .venv
.venv/Scripts/pip install -r requirements-lock.txt   # Windows development deps
pytest -q                                            # V0 suite + V1 unit tests
python scripts/v1/00_check_environment.py --device cpu
```

PyTorch/PyG are intentionally not installed here; `00_check_environment.py
--device cpu` reports them as unavailable and still passes, so the scaffold
can be verified before the deep stack is present.

## 3. GPU server environment

Authoritative V1 training platform: Linux, Python 3.11, 1 x NVIDIA A100
(40 GB or 80 GB), 64 GB RAM recommended. Inspect the server driver first
(`nvidia-smi`), then install the curated stack:

```bash
python3.11 -m venv .venv
.venv/bin/pip install -r requirements-v1-lock-linux.txt
.venv/bin/pip install -e ".[dev]"
.venv/bin/python scripts/v1/00_check_environment.py --device cuda
pytest -q
```

The lock file pins the selected official combination
(PyTorch 2.11.0 + cu128, PyG 2.8.0, pyg-lib 0.7.0). After validation on the
actual server, replace the curated list with a full resolved `pip freeze`
output recorded on that server; never copy a Windows lock file.

## 4. Reproducibility contract (applies from Milestone 0)

- Every run accepts `--seed`, `--device`, `--fold`, `--config`, `--resume`,
  and `--output-dir` where applicable.
- Deterministic algorithms are enabled where supported; any nondeterministic
  CUDA operation is recorded in QC.
- AMP is used only after an FP32 equivalence smoke test.
- Checkpoints carry the manifest schema from
  `src/phenotype_network_v1/checkpoint.py`: schema version, created time,
  seed, epoch, config hash, data hashes, Git commit, and framework versions.
- Checkpoints, raw data, patient data, and generated outputs are never
  committed (`outputs/v1/`, `data/processed/v1/`, `data/graphs/v1/`,
  `checkpoints/`, `*.pt`, `*.pth` are ignored).
- All filesystem paths use `pathlib.Path` and honour the existing
  `PHENOTYPE_NETWORK_*_DIR` environment variables.
- Structured logging is required in application code.

## 5. First server validation checklist

1. `nvidia-smi`: record driver and CUDA versions.
2. Install `requirements-v1-lock-linux.txt`.
3. Run `00_check_environment.py --device cuda`: record PyTorch, PyG, CUDA,
   cuDNN, GPU model, deterministic settings, and free memory.
4. Run `pytest -q` including `gpu`-marked tests.
5. Freeze the resolved environment back into `requirements-v1-lock-linux.txt`
   and commit the update with the environment report hash.
