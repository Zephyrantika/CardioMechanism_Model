# Server Migration

V0 is CPU-only and uses Python 3.11, SciPy sparse matrices, Parquet, and Leiden. No GPU, deep-learning framework, or platform-specific path is required.

## Capacity

The current workspace uses about 0.74 GiB under `data/` and 0.68 GiB under `outputs/` before final path outputs. Allocate at least 10 GiB disk for temporary files and reruns. Milestone 11 peaks near 1.7 GiB private memory on the current data; 8 GiB RAM is the minimum and 16 GiB is recommended. More cores do not speed up the current serial scripts, but folds can be scheduled as separate future jobs after adding checkpoint orchestration.

## Transfer

Transfer the Git repository plus the ignored `data/raw/` files and, when retaining current results, `data/processed/`, `data/folds/`, `data/graphs/`, and `outputs/`. Verify the frozen external files against `data/raw/external_validation_manifest.json` and `outputs/qc/external_validation_qc.json`. Never merge validation resources into graph inputs.

Example with `rsync` from a Unix-like client:

```bash
rsync -av --exclude .venv --exclude __pycache__ phenotype-network-v0/ user@server:/srv/phenotype-network-v0/
```

## Environment

```bash
cd /srv/phenotype-network-v0
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-lock.txt
python -m pip install -e . --no-deps
pytest -q
```

The lock contains only version pins, not Windows wheel paths. If a pinned wheel is unavailable for the server architecture, build in a clean Python 3.11 environment and record the resulting platform-specific lock separately.

## Storage Overrides

Large directories can live on mounted storage without editing configuration:

```bash
export PHENOTYPE_NETWORK_RAW_DIR=/mnt/project/raw
export PHENOTYPE_NETWORK_INTERIM_DIR=/mnt/project/interim
export PHENOTYPE_NETWORK_PROCESSED_DIR=/mnt/project/processed
export PHENOTYPE_NETWORK_FOLDS_DIR=/mnt/project/folds
export PHENOTYPE_NETWORK_GRAPHS_DIR=/mnt/project/graphs
export PHENOTYPE_NETWORK_OUTPUTS_DIR=/mnt/project/outputs
```

All scripts use `configs/default.yaml` and `pathlib.Path`. Run commands from the repository root. Use `--overwrite` only for derived interim, processed, graph, or output files; raw files are immutable.

## V0 Final Stages

```bash
python scripts/10_prepare_external_validation.py --config configs/default.yaml
python scripts/10_run_ablations.py --config configs/default.yaml
python scripts/10_evaluate_and_validate.py --config configs/default.yaml
python scripts/11_detect_modules.py --config configs/default.yaml
python scripts/12_build_explanations.py --config configs/default.yaml
python scripts/13_build_case_reports.py --config configs/default.yaml
```

Use a batch allocation of at least two hours for Milestone 11 and three hours for Milestone 12 on hardware comparable to the current workstation. Preserve logs and QC JSON files with the outputs.