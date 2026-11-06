# Agent instructions

This repository contains a computationally expensive Bayesian evaluation
pipeline. Follow the documented commands exactly and do not infer hidden
defaults from implementation modules.

## Supported interfaces

CLI:

```bash
lm-bmfrm-evaluate --config CONFIG.json
```

Python:

```python
from lm_bmfrm import load_experiment, run_experiment
```

The CLI and Python interface use the same Stage 1–3 orchestration path.

## Before running

1. Work from the repository root.
2. Install dependencies with `pip install -r requirements.txt`, then install
   the package with `pip install -e .`. Use Python 3.11 or 3.12; Python 3.11
   is recommended for reproducing the released environment.
3. Read the selected JSON configuration.
4. Preserve the order of `data.judge_files`.
5. Do not modify seeds, file ordering, dimensions, sampling settings, or
   thresholds when reproducing an archived experiment.
6. Expect CPU execution when no compatible JAX GPU is available; it may be
   considerably slower.

`requirements.txt` is the authoritative dependency list, including the JAX
CUDA 12 packages. `pyproject.toml` loads its dependencies from that file.

## Input contract

Judge files are Parquet tables. Required columns:

- `item_id`
- `system_id`
- `rater_id`
- `dimension`
- `score`

`score` must be an integer from 1 through 5. An optional `score_list` column may
contain repeated ordinal ratings. See `INPUT_SCHEMA.md`.

## Reproduce an archived experiment

Each released result directory contains `resolved_config.json`. Run:

```bash
lm-bmfrm-evaluate \
  --config results/EXPERIMENT/resolved_config.json
```

The configuration contains the effective seeds, thresholds, dimensions,
sampling settings, and ordered judge-file paths used for that experiment.

## Read results without refitting

```python
from lm_bmfrm import load_experiment

result = load_experiment("results/EXPERIMENT")
print(result.leaderboard)
```

## Successful completion

Do not parse formatted terminal tables. A successful experiment directory must
contain:

- `resolved_config.json`
- `run_config.json`
- `judge_input_manifest.parquet`
- `leaderboard_results.parquet`
- `predictions.parquet`

Use:

```python
result.machine_summary()
```

to obtain a JSON-serializable completion record.

## Primary outputs

- `leaderboard_results.parquet`: claim-specific judge metrics
- `predictions.parquet`: held-out system predictions
- `human_audit.parquet`: Stage-1 diagnostics
- `llm_audit.parquet`: Stage-2 diagnostics
- `benchmark_cards.parquet`: benchmark-resolution summaries
- `rhat_records.parquet`: sampling diagnostics

See `OUTPUT_SCHEMA.md` for the complete map.

## Safety rules

- Do not edit functions protected by `numerical_contract.json`.
- Do not reorder judge files.
- Do not recompute reported values manually from console output.
- Do not treat the example configuration as a scientifically adequate run.
- Do not overwrite archived result directories.
