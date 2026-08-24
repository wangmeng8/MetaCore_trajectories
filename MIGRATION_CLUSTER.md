# Cluster Migration Notes

This archive is intended for moving MetaCoreBench from the Windows workstation to a Linux cluster.

## What Is Included

- Project code: `scripts/`, `tests/`, `README.md`, `pyproject.toml`
- Environment templates: `.env.example`
- Dependency snapshots: `requirements-freeze-main.txt`, `requirements-freeze-hle.txt`
- Local benchmark checkouts/data: `.external/`, `data/`

## What Is Not Included

- `.env` and API keys
- Windows virtual environments: `.venv/`, `.venv-hle/`
- Nested virtual environments inside downloaded benchmark repositories, such as `.external/*/.venv/`
- Local run outputs: `outputs/`, `jobs/`, `logs/`
- Git metadata directories

## Recreate This Archive

Use the packaging script instead of hand-written `tar --exclude` commands:

```bash
python scripts/create_migration_package.py
```

The script recursively excludes virtual environments, API-key files, local outputs, Git metadata, and cache directories even when they are nested under `.external/`.

Windows virtual environments generally cannot run on a Linux cluster. Recreate the venv on the cluster:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -e .
python -m pip install tau2-bench
python -m pip install -r .external/OSWorld/requirements.txt
python -m pip install pandas pyarrow modelscope
```

If the cluster runs Tau2 only through the cloned checkout, install `uv` and configure:

```bash
python -m pip install uv
export TAU2_COMMAND="uv run tau2"
export TAU2_REPO_DIR=".external/tau2-bench"
```

Then verify:

```bash
python scripts/run_tau2_gpt55.py --dry-run
```

Then create `.env` from `.env.example` and set your API key/base URL.

## Quick Checks

```bash
source .venv/bin/activate
python scripts/run_hle_tools.py --dry-run
python scripts/run_tau2_gpt55.py --dry-run
python scripts/collect_trajectories.py --help
python -m unittest discover -s tests
```

Terminal-Bench and OSWorld also require Docker or another supported sandbox provider on the cluster.
