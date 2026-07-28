# Experiment configuration

`paper_default.json` contains the historical paper-run defaults that were previously
embedded in the evaluation CLI. Loading this file does not change the numerical
settings, model choice, random seeds, call order, or output schema.

Run with the packaged defaults and explicit benchmark selection:

```bash
python scripts/run_evaluation.py --benchmark summeval
```

Run with a checked-in configuration:

```bash
python scripts/run_evaluation.py --config configs/summeval_paper.json
```

A user configuration is recursively merged over `paper_default.json`. Unknown keys,
invalid probability ranges, duplicate file entries, and invalid dimensions are
rejected before probabilistic fitting begins. Command-line arguments override the
merged configuration.

For an auditable paper run, set `data.judge_files` explicitly in the desired order.
That order is preserved because it can affect category factorization and serialized
row order. When `judge_files` is omitted, the evaluator deliberately retains the
legacy `glob.glob` discovery order rather than silently sorting files and potentially
changing previous results.

Every run writes:

- `resolved_config.json`: the complete effective configuration;
- `judge_input_manifest.parquet`: input order, path, SHA-256, size, and row count;
- `run_config.json`: command line, package versions, Git state, and runtime metadata.

Relative paths are resolved from the working directory used to launch the command.
