# Output artifacts

Each evaluation creates one timestamped directory under `outputs.results_root`.

## Reproducibility and provenance

| File | Purpose |
|---|---|
| `resolved_config.json` | Complete effective configuration, including seeds and ordered inputs |
| `run_config.json` | Command, package versions, Git state, and runtime metadata |
| `judge_input_manifest.parquet` | Ordered paths, hashes, sizes, and row counts |
| `judge_coverage.csv` | Judge coverage against the benchmark grid |
| `judge_exclusions.csv` | Input rows/files excluded during preparation |

## Stage 1

| File | Purpose |
|---|---|
| `human_audit.parquet` | Human-rater fit diagnostics |
| `human_item_audit.parquet` | Human item diagnostics |
| `benchmark_cards.parquet` | Benchmark-resolution summaries |
| `benchmark_curves.parquet` | Information and precision curves |
| `stage1_params_<dimension>.json` | Stage-1 posterior summaries |

## Stage 2

| File | Purpose |
|---|---|
| `llm_audit.parquet` | Judge-level fit diagnostics |
| `llm_item_audit.parquet` | Judge-by-item diagnostics |
| `dif_audit_<dimension>_<judge>.parquet` | Judge-by-item DIF summaries

## Stage 3

| File | Purpose |
|---|---|
| `predictions.parquet` | Held-out system estimates and uncertainty |
| `leaderboard_results.parquet` | Rank, pairwise, distributional, interval, and support metrics |
| `q_raw_loocv.json` | Raw conformal quantiles |
| `q_norm_loocv.json` | Normalized conformal quantiles |
| `rhat_records.parquet` | Sampling diagnostics |
| `loocv_posterior_artifacts.npz` | Persisted posterior arrays |
| `loocv_posterior_artifacts_manifest.parquet` | Posterior-artifact metadata |

The primary downstream result is `leaderboard_results.parquet`. Programs should
use column names from the file rather than relying on rendered console tables.