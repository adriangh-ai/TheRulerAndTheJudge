"""Input validation, coverage accounting, and judge-file provenance."""
from __future__ import annotations

from collections import Counter
import glob
import hashlib
import os
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd


REQUIRED_JUDGMENT_COLUMNS = ("item_id", "system_id", "rater_id", "dimension", "score")
IDENTIFIER_COLUMNS = ("item_id", "system_id", "rater_id", "dimension")


def safe_mode(x):
    """Return the lowest modal ordinal score from a repeated score list."""
    if (not isinstance(x, list) and not isinstance(x, np.ndarray)) or len(x) == 0:
        return np.nan
    vals = [int(v) for v in x if pd.notna(v)]
    if len(vals) == 0:
        return np.nan
    counts = Counter(vals)
    max_count = max(counts.values())
    modes = [k for k, count in counts.items() if count == max_count]
    return min(modes)


def validate_score_lists(series: pd.Series, *, source: str, n_classes: int = 5) -> None:
    """Validate every non-missing repeated score before deriving its modal label."""
    for row_index, value in series.items():
        if not isinstance(value, (list, np.ndarray)) or len(value) == 0:
            raise ValueError(f"{source} has an invalid/empty score_list at row {row_index}.")
        for score in value:
            if pd.isna(score):
                continue
            try:
                numeric = float(score)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"{source} has a non-numeric score_list value at row {row_index}: {score!r}"
                ) from exc
            if not np.isclose(numeric, np.rint(numeric)) or not 1 <= numeric <= n_classes:
                raise ValueError(
                    f"{source} has an invalid score_list value at row {row_index}: {score!r}"
                )


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve_judge_files(
    judge_folder: str,
    benchmark: str,
    explicit_files: Sequence[str] | None = None,
) -> list[str]:
    """Resolve judge parquet files without changing their declared/discovered order.

    Explicit file lists are the deterministic paper-run mechanism. When no list is
    supplied, the historical ``glob.glob`` discovery order is retained exactly so
    existing result ordering is not silently changed.
    """
    if explicit_files is not None:
        judge_files = [str(Path(path)) for path in explicit_files]
    else:
        benchmark_folder = os.path.join(judge_folder, benchmark)
        judge_files = glob.glob(os.path.join(benchmark_folder, "*.parquet"))

    if not judge_files:
        location = ", ".join(explicit_files) if explicit_files else os.path.join(judge_folder, benchmark)
        raise FileNotFoundError(f"No judge parquet files found for benchmark '{benchmark}' in {location}.")

    duplicates = [path for path, count in Counter(judge_files).items() if count > 1]
    if duplicates:
        raise ValueError(f"Judge file list contains duplicates: {duplicates}")

    missing = [path for path in judge_files if not Path(path).is_file()]
    if missing:
        raise FileNotFoundError(f"Judge file(s) not found: {missing}")

    return judge_files


def validate_judgment_dataframe(
    df: pd.DataFrame,
    *,
    source: str,
    n_classes: int = 5,
) -> None:
    """Validate the common judgment schema without modifying row order or values."""
    missing_columns = [column for column in REQUIRED_JUDGMENT_COLUMNS if column not in df.columns]
    if missing_columns:
        raise ValueError(f"{source} is missing required columns: {missing_columns}")

    empty_identifier = pd.Series(False, index=df.index)
    for column in IDENTIFIER_COLUMNS:
        empty_identifier |= df[column].isna() | df[column].astype(str).str.strip().eq("")
    if empty_identifier.any():
        bad_rows = df.index[empty_identifier].tolist()[:10]
        raise ValueError(f"{source} has missing/empty identifiers at row(s): {bad_rows}")

    numeric_scores = pd.to_numeric(df["score"], errors="coerce")
    invalid_numeric = numeric_scores.isna()
    if invalid_numeric.any():
        bad_rows = df.index[invalid_numeric].tolist()[:10]
        raise ValueError(f"{source} has non-numeric scores at row(s): {bad_rows}")

    non_integral = ~np.isclose(numeric_scores.to_numpy(dtype=float), np.rint(numeric_scores))
    if non_integral.any():
        bad_rows = df.index[np.flatnonzero(non_integral)].tolist()[:10]
        raise ValueError(f"{source} has non-integral ordinal scores at row(s): {bad_rows}")

    out_of_range = (numeric_scores < 1) | (numeric_scores > n_classes)
    if out_of_range.any():
        bad_values = sorted(numeric_scores[out_of_range].unique().tolist())
        raise ValueError(
            f"{source} has scores outside [1, {n_classes}]: {bad_values}"
        )


def load_judge_outputs(
    judge_files: Sequence[str],
    *,
    n_classes: int = 5,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load judge outputs in the supplied order and return audit manifests.

    Rows and files are concatenated in exactly the incoming order. The returned
    file manifest records that order together with content hashes, allowing a
    future paper run to reproduce the same factorization/order contract by using
    ``data.judge_files`` in the configuration.
    """
    frames: list[pd.DataFrame] = []
    exclusion_rows: list[dict] = []
    manifest_rows: list[dict] = []

    for position, file_path in enumerate(judge_files):
        frame = pd.read_parquet(file_path).copy()

        if "generated_score" in frame.columns:
            if "score" in frame.columns:
                raise ValueError(
                    f"File {file_path} contains both 'generated_score' and 'score'; "
                    "remove the ambiguous duplicate before evaluation."
                )
            frame = frame.rename(columns={"generated_score": "score"})

        if "score_list" in frame.columns:
            validate_score_lists(
                frame["score_list"], source=f"judge file {file_path}", n_classes=n_classes
            )
            score_from_list = frame["score_list"].apply(safe_mode)
            if score_from_list.isna().any():
                bad_rows = frame.index[score_from_list.isna()].tolist()[:10]
                raise ValueError(
                    f"File {file_path} has invalid or empty score_list values at row(s): {bad_rows}"
                )
            # Repeated generations are represented by their full distribution in
            # Stage 2; the modal hard score is retained for classical diagnostics.
            frame["score"] = score_from_list

        n_before = len(frame)
        if "score" in frame.columns:
            frame = frame.dropna(subset=["score"])

        validate_judgment_dataframe(frame, source=f"judge file {file_path}", n_classes=n_classes)

        exclusion_rows.append(
            {
                "file": os.path.basename(file_path),
                "n_rows_loaded": int(n_before),
                "n_rows_after_dropna": int(len(frame)),
                "n_dropped_nan_score": int(n_before - len(frame)),
            }
        )
        manifest_rows.append(
            {
                "position": int(position),
                "path": str(file_path),
                "filename": os.path.basename(file_path),
                "sha256": _sha256(file_path),
                "size_bytes": int(Path(file_path).stat().st_size),
                "n_rows": int(len(frame)),
            }
        )
        frames.append(frame)

    return (
        pd.concat(frames, ignore_index=True),
        pd.DataFrame(exclusion_rows),
        pd.DataFrame(manifest_rows),
    )


def resolve_dimensions(
    judge_df: pd.DataFrame,
    benchmark: str,
    configured_dimensions: Iterable[str] | None,
) -> list[str]:
    """Resolve dimension order while preserving the historical first-seen order."""
    observed = [str(value) for value in judge_df["dimension"].drop_duplicates().tolist()]

    if benchmark == "summeval_synth":
        dimensions = ["coherence"]
    elif configured_dimensions is None:
        dimensions = observed
    else:
        dimensions = [str(value) for value in configured_dimensions]

    missing = [dimension for dimension in dimensions if dimension not in observed]
    if missing:
        raise ValueError(f"Configured dimensions are absent from judge outputs: {missing}")
    if not dimensions:
        raise ValueError("No evaluation dimensions were resolved.")
    return dimensions


def build_judge_coverage_table(j_df, df_human, dims):
    """Compute coverage against the human system-by-item grid for each judge."""
    rows = []
    for dimension in dims:
        human_dimension = df_human[df_human["dimension"] == dimension]
        expected_cells = int(
            human_dimension[["system_id", "item_id"]].drop_duplicates().shape[0]
        )
        judge_dimension = j_df[j_df["dimension"] == dimension]
        for judge_id, subset in judge_dimension.groupby("rater_id"):
            observed = int(subset[["system_id", "item_id"]].drop_duplicates().shape[0])
            n_rows = int(len(subset))
            rows.append(
                {
                    "dimension": dimension,
                    "judge_model": str(judge_id),
                    "expected_cells": expected_cells,
                    "observed_cells": observed,
                    "n_rows": n_rows,
                    "missing_cells": expected_cells - observed,
                    "coverage_frac": (observed / expected_cells) if expected_cells > 0 else np.nan,
                }
            )
    return pd.DataFrame(rows)
