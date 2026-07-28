"""Prediction and result artifact serialization."""
from __future__ import annotations
import os
import numpy as np
import pandas as pd
from .loocv import run_system_cv_conformal


def _jsonify_qdict(qdict):
    out = {}
    for outer, dim_dict in qdict.items():
        out[str(outer)] = {}
        for dim, judge_dict in dim_dict.items():
            out[str(outer)][str(dim)] = {str(j): float(v) for j, v in judge_dict.items()}
    return out


def _jsonify_q_flat(qdict):
    """flat (dim -> judge -> q) JSON for LOOCV-mode conformal quantiles."""
    return {str(dim): {str(j): float(v) for j, v in jd.items()} for dim, jd in qdict.items()}


def save_prediction_artifacts(nested_results, results_dir, prefix="nested"):
    """Persist posterior vectors/cutpoints separately from parquet tables."""
    artifact_arrays = nested_results.get("artifact_arrays", {})
    if artifact_arrays:
        np.savez_compressed(
            os.path.join(results_dir, f"{prefix}_posterior_artifacts.npz"),
            **artifact_arrays,
        )
    artifact_manifest = nested_results.get("artifact_manifest", pd.DataFrame())
    if isinstance(artifact_manifest, pd.DataFrame) and not artifact_manifest.empty:
        artifact_manifest.to_parquet(
            os.path.join(results_dir, f"{prefix}_posterior_artifacts_manifest.parquet"),
            index=False,
        )


def parquet_safe_predictions(df):
    """Drop heavy object-array columns; the NPZ artifact file stores them by key."""
    if df is None or df.empty:
        return pd.DataFrame()
    drop_cols = [c for c in ["post_theta", "post_human", "theta_grid"] if c in df.columns]
    return df.drop(columns=drop_cols)


def run_judge_benchmark(
    judge_df,
    human_df,
    dims,
    idata_anchor_by_dim,
    theta_grid,
    n_total_draws=4000,
    n_splits=5,
    target_accept=0.99,
    anchor_draw_selection_method="medoids",
    anchor_draw_compute_n=64,
    anchor_draw_fit_n=None,
    save_artifacts=True,
    alpha=0.05,
    chains_cv=4,
    use_mc=True,
    add_alpha_main=False,
    sampler_seed=42,
    anchor_draw_compute_seed=42,
    anchor_draw_fit_seed=123,
    kfold_random_state=42,
):
    pm_inputs = pd.concat([human_df, judge_df], ignore_index=True)
    return run_system_cv_conformal(
        pm_inputs=pm_inputs,
        dims=dims,
        idata_anchor_by_dim=idata_anchor_by_dim,
        theta_grid=theta_grid,
        n_splits=n_splits,
        alpha=alpha,
        draws_cv=n_total_draws,
        chains_cv=chains_cv,
        target_accept=target_accept,
        anchor_draw_selection_method=anchor_draw_selection_method,
        anchor_draw_compute_n=anchor_draw_compute_n,
        anchor_draw_fit_n=anchor_draw_fit_n,
        save_artifacts=save_artifacts,
        use_mc=use_mc,
        add_alpha_main=add_alpha_main,
        sampler_seed=sampler_seed,
        anchor_draw_compute_seed=anchor_draw_compute_seed,
        anchor_draw_fit_seed=anchor_draw_fit_seed,
        kfold_random_state=kfold_random_state,
    )
