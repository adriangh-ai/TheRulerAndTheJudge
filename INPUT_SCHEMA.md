# Judge-input schema

The evaluation pipeline consumes one or more Parquet files containing judge
ratings. One file may contain one judge or multiple judge identifiers.

## Required columns

| Column | Type | Description |
|---|---|---|
| `item_id` | string-compatible | Identifier matching an item in the selected human benchmark |
| `system_id` | string-compatible | Evaluated system identifier |
| `rater_id` | string-compatible | Judge/model identifier |
| `dimension` | string-compatible | Evaluation dimension |
| `score` | integer | Ordinal score in `{1, 2, 3, 4, 5}` |

## Optional columns

| Column | Type | Description |
|---|---|---|
| `score_list` | list of integers | Repeated ratings for the same row |
| `model_output` | string | Raw judge response |
| `prompt` | string | Prompt supplied to the judge |
| `is_llm` | boolean | Accepted but overwritten as `True` by the evaluator |

When `score_list` is supplied, each valid row represents one empirical
distributional observation. Its values must also lie in `{1, 2, 3, 4, 5}`.

## Identity requirements

Judge `item_id`, `system_id`, and `dimension` values must correspond to the
selected benchmark. Missing judge coverage is reported in
`judge_coverage.csv`; dropped or invalid rows are reported in
`judge_exclusions.csv`.

## Deterministic ordering

For reproducible runs, list files explicitly in `data.judge_files`. Their order
is preserved and recorded in `judge_input_manifest.parquet`.

## Minimal validation

```python
import pandas as pd

path = "examples/custom_judge/custom_judge.parquet"
frame = pd.read_parquet(path)

required = {
    "item_id",
    "system_id",
    "rater_id",
    "dimension",
    "score",
}
missing = required - set(frame.columns)
if missing:
    raise ValueError(f"Missing columns: {sorted(missing)}")

if not frame["score"].dropna().astype(int).between(1, 5).all():
    raise ValueError("Scores must be integers from 1 through 5.")