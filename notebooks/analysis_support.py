from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from lm_bmfrm import load_experiment

NOTEBOOK_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = NOTEBOOK_DIR.parent
RUNS_FILE = NOTEBOOK_DIR / "runs.json"
RUNS = {name: (RUNS_FILE.parent / path).resolve()
        for name, path in json.loads(RUNS_FILE.read_text()).items()}


def open_results():
    """Validate the pinned completed archives without fitting or rewriting them."""
    results = {}
    for dataset, folder in RUNS.items():
        result = load_experiment(folder)
        if not (folder / "judge_input_manifest.parquet").is_file():
            raise FileNotFoundError(folder / "judge_input_manifest.parquet")
        config = result.resolved_config
        if config["evaluation"]["cv_mode"] != "loocv":
            raise ValueError(f"{dataset}: these tables require a LOOCV experiment")
        results[dataset] = result
    return results


def output_dir(notebook):
    if notebook not in {"final_analysis", "rebuttal_supporting_tables"}:
        raise ValueError(notebook)
    path = NOTEBOOK_DIR / "outputs" / notebook
    path.mkdir(parents=True, exist_ok=True)
    return path


def leaderboard(dataset):
    frame = load_experiment(RUNS[dataset]).leaderboard.copy()
    if "Eval split" in frame:
        frame = frame[frame["Eval split"].astype(str).str.casefold().eq("loocv")]
    if "Outer fold" in frame:
        frame = frame[pd.to_numeric(frame["Outer fold"], errors="raise").eq(0)]
    if frame.duplicated(["Dimension", "Model"]).any():
        raise ValueError(f"{dataset}: duplicate LOOCV dimension/model rows")

    # For older archives that used "NFSM" instead of "NSMF" in column names, normalize the naming
    # to match the old visualization scripts.
    # Keep the saved columns and add the names consumed by the paper notebook.
    for column in list(frame.columns):
        prefix = str(column).split(" ", 1)[0]
        if prefix.casefold() not in {"nsmf", "nfsm"}:
            continue
        canonical = "NFSM" + column[len(prefix):]
        if canonical == column:
            continue
        if canonical in frame:
            overlap = frame[canonical].notna() & frame[column].notna()
            if not np.allclose(
                frame.loc[overlap, canonical], frame.loc[overlap, column],
                rtol=0, atol=1e-12,
            ):
                raise ValueError(f"{dataset}: conflicting metric columns {canonical!r} and {column!r}")
            frame[canonical] = frame[canonical].combine_first(frame[column])
        else:
            frame[canonical] = frame[column]

    # Older archives save the exact support confusion counts, but not these names.
    if "FSR numerator" not in frame and "Support FP" in frame:
        frame["FSR numerator"] = frame["Support FP"]
    if "Human unsupported pairs" not in frame and {"Support FP", "Support TN"} <= set(frame):
        frame["Human unsupported pairs"] = frame["Support FP"] + frame["Support TN"]
    return frame.copy()


def _comparison_distribution(coordinates, masses):
    """Validate and normalize one saved discrete distribution."""
    coordinates = np.asarray(coordinates, dtype=np.float64)
    masses = np.asarray(masses, dtype=np.float64)
    if (
        coordinates.ndim != 1 or coordinates.size == 0
        or coordinates.shape != masses.shape
        or not np.isfinite(coordinates).all() or not np.isfinite(masses).all()
        or np.any(np.diff(coordinates) <= 0) or np.any(masses < 0)
        or not np.isfinite(masses.sum()) or masses.sum() <= 0
    ):
        raise ValueError("Invalid comparison coordinates or posterior masses")
    return coordinates, masses / masses.sum()


def _dominance_components(a, b):
    """Return strict dominance and exact tie mass on the saved coordinates."""
    coordinates_a, masses_a = a
    coordinates_b, masses_b = b
    below = np.searchsorted(coordinates_b, coordinates_a, side="left")
    cdf_b = np.concatenate(([0.0], np.cumsum(masses_b)))
    p_gt = float(np.dot(masses_a, cdf_b[below]))
    in_bounds = below < len(coordinates_b)
    a_indices = np.flatnonzero(in_bounds)
    a_indices = a_indices[coordinates_a[a_indices] == coordinates_b[below[a_indices]]]
    p_equal = float(np.dot(masses_a[a_indices], masses_b[below[a_indices]]))
    return p_gt, p_equal


def pairwise_table(dataset, folder):
    """Use saved coordinates, including historical shared grids and mapped Hanna.

    Exact coordinate equality determines ties. Equal indices on different supports
    are not ties. The public reader validates metadata and reference rulers.
    """
    predictions = load_experiment(folder).comparison_predictions()
    rows = []
    for (dimension, model), group in predictions.groupby(
        ["dimension", "judge_model"], sort=True
    ):
        group = group.copy()
        group["system_id"] = group["system_id"].astype(str)
        if group.duplicated("system_id").any():
            raise ValueError(f"{dataset}/{dimension}/{model}: duplicate systems")
        for column in ("comparison_reference", "comparison_representation"):
            if column in group and group[column].nunique(dropna=False) != 1:
                raise ValueError(f"{dataset}/{dimension}/{model}: conflicting {column}")
        human, judge = {}, {}
        for rec in group.itertuples(index=False):
            human[rec.system_id] = _comparison_distribution(
                rec.post_human_coordinates, rec.post_human
            )
            judge[rec.system_id] = _comparison_distribution(
                rec.post_theta_coordinates, rec.post_theta
            )
        for a, b in combinations(sorted(human), 2):
            h_gt, h_eq = _dominance_components(human[a], human[b])
            j_gt, j_eq = _dominance_components(judge[a], judge[b])
            rows.append({
                "dataset": dataset,
                "dimension": dimension,
                "judge_model": model,
                "_dimension_key": str(dimension).strip().casefold(),
                "_model_key": str(model).strip().casefold(),
                "system_a": a,
                "system_b": b,
                "human_p_gt_strict": h_gt,
                "human_p_equal_grid": h_eq,  # Historical column name; exact coordinates.
                "human_p_mid": h_gt + 0.5 * h_eq,
                "human_signed_margin": 2 * (h_gt + 0.5 * h_eq) - 1,
                "judge_p_gt_strict": j_gt,
                "judge_p_equal_grid": j_eq,
                "judge_p_mid": j_gt + 0.5 * j_eq,
                "judge_signed_margin": 2 * (j_gt + 0.5 * j_eq) - 1,
            })
    return pd.DataFrame(rows)


def criterion_winners(group, metric, fsr_max=None):
    """Keep every tied winner; selection never depends on source row order."""
    valid = group.dropna(subset=[metric]).copy()
    if fsr_max is not None:
        valid = valid[valid["False Separation Rate"].le(fsr_max)]
    if valid.empty:
        return valid
    return valid[np.isclose(valid[metric], valid[metric].max(), rtol=0, atol=1e-12)]


def hanna_criterion_audit(fsr_max=0.30):
    """All six dimensions, with ties and all three Stage-3 criteria visible."""
    rows = []
    for dimension, group in leaderboard("HANNA").groupby("Dimension", sort=True):
        selected = {}
        for criterion, metric, tolerance in (
            ("Rank", "Spearman", None),
            ("nSMF", "NFSM", None),
            ("FSR", "NFSM", fsr_max),
        ):
            winners = criterion_winners(group, metric, tolerance)
            selected[criterion] = set(winners["Model"].astype(str))
            for _, row in winners.iterrows():
                rows.append({
                    "Dataset": "HANNA", "Dimension": dimension,
                    "Criterion": criterion, "Model": row["Model"],
                    "Spearman": row["Spearman"], "NFSM": row["NFSM"],
                    "FSR": row["False Separation Rate"],
                    "FSR numerator": row["FSR numerator"],
                    "Human unsupported pairs": row["Human unsupported pairs"],
                })
            if winners.empty:
                rows.append({"Dataset": "HANNA", "Dimension": dimension,
                             "Criterion": criterion, "Model": "ABSTAIN"})
        disagreement = len({frozenset(v) for v in selected.values()}) > 1
        for row in rows:
            if row["Dimension"] == dimension:
                row["Criteria disagree"] = disagreement
    return pd.DataFrame(rows)
