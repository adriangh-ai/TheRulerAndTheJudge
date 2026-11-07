# Programmatic usage

The public Python interface delegates to the same configuration-driven
orchestration function used by `lm-bmfrm-evaluate`. It does not provide an
alternative numerical implementation.

## Run an experiment

Run commands from the repository root so that relative input paths in the
configuration resolve consistently.

```python
import json

from lm_bmfrm import run_experiment

result = run_experiment(
    "examples/custom_judge/config.json",
    results_root="results/programmatic",
)

print(json.dumps(result.machine_summary(), indent=2))
```

`run_experiment` returns only after the complete Stage 1–3 pipeline finishes.
Exceptions and execution failures are propagated to the caller.

## Read an existing experiment

```python
from lm_bmfrm import load_experiment

result = load_experiment(
    "results/hanna_MC_per_judge_linspace_20260726_200116"
)

leaderboard = result.leaderboard
predictions = result.predictions

print(
    leaderboard[
        ["Dimension", "Model", "Spearman", "NSMF", "False Separation Rate"]
    ]
)
```

## Read additional outputs

```python
human_audit = result.read_table("human_audit")
llm_audit = result.read_table("llm_audit")
benchmark_cards = result.read_table("benchmark_cards")
provenance = result.provenance
resolved_config = result.resolved_config
```

The documented table names are:

- `leaderboard`
- `predictions`
- `human_audit`
- `human_item_audit`
- `llm_audit`
- `llm_item_audit`
- `benchmark_cards`
- `benchmark_curves`
- `rhat_records`

## Automation contract

A successful run:

1. returns without raising an exception;
2. creates exactly one timestamped experiment directory;
3. writes `resolved_config.json` and `run_config.json`;
4. writes `judge_input_manifest.parquet`;
5. writes `leaderboard_results.parquet` and `predictions.parquet`.

Agents and downstream programs should read these files rather than parse
progress bars or formatted console tables.
