"""Shared cross-validation helpers and artifact primitives."""
from __future__ import annotations
import arviz as az
import numpy as np
from .provenance import _sampler_diagnostics


def _safe_key_part(x):
    """Make a short filesystem/npz-safe key component."""
    s = str(x)
    for ch in ["/", "\\", " ", ":", ";", ",", "=", "|", "\n", "\t"]:
        s = s.replace(ch, "__")
    return s


def _posterior_mean_sd(post, theta_grid):
    post = np.asarray(post, dtype=np.float64)
    theta_grid = np.asarray(theta_grid, dtype=np.float64)
    post = post / np.clip(post.sum(), 1e-15, None)
    mean = float((post * theta_grid).sum())
    sd = float(np.sqrt((post * (theta_grid - mean) ** 2).sum()))
    return mean, sd


def _extract_full_human_theta_by_dim(human_anchor_by_dim):
    """
    Extract full-data human theta maps from the Stage-1 audit output.

    Expected input is audit_results["human_anchor"], whose values are the
    full-data human anchor fits returned by fit_fold_models_per_dim(...)["human"].
    """
    if human_anchor_by_dim is None:
        return None
    out = {}
    for dim, anchor in human_anchor_by_dim.items():
        if isinstance(anchor, dict) and "human_theta_map" in anchor:
            out[str(dim)] = {str(k): float(v) for k, v in anchor["human_theta_map"].items()}
        elif isinstance(anchor, dict) and "fold_theta_human_map" in anchor:
            out[str(dim)] = {str(k): float(v) for k, v in anchor["fold_theta_human_map"].items()}
        elif isinstance(anchor, dict):
            out[str(dim)] = {str(k): float(v) for k, v in anchor.items()}
    return out


def _extract_full_human_pairwise_by_dim(human_anchor_by_dim, theta_var_name="theta"):
    """
    Extract full-data human posterior dominance probabilities per dimension.

    The resulting maps are used only for outer split design diagnostics/constraints,
    never for fitting a judge, estimating cutpoints, calibrating conformal q, or
    computing held-out predictions.
    """
    if human_anchor_by_dim is None:
        return None

    out = {}
    for dim, anchor in human_anchor_by_dim.items():
        if not isinstance(anchor, dict) or "idata_anchor" not in anchor:
            continue
        idata = anchor["idata_anchor"]
        post = idata.posterior
        if theta_var_name not in post:
            continue
        systems = [str(x) for x in post.coords["system"].values]
        theta_draws = np.asarray(post[theta_var_name]).reshape(-1, len(systems))
        pair_map = {}
        for i, a in enumerate(systems):
            for j in range(i + 1, len(systems)):
                b = systems[j]
                # Dominance probability Pr(theta_a > theta_b), matching the
                # support rule used for pairwise decisions.
                pair_map[(str(a), str(b))] = float(np.mean(theta_draws[:, i] > theta_draws[:, j]))
        out[str(dim)] = pair_map
    return out


def _extract_rhat_records(idata_llm, outer_fold, eval_split, dimension, judge_model, inner_fold=None):
    rhat_df = (
        az.rhat(idata_llm)
        .to_array()
        .to_series()
        .dropna()
        .sort_values(ascending=False)
        .reset_index(name="r_hat")
    )
    # divergence count and minimum ESS alongside R-hat.
    n_div, min_ess = _sampler_diagnostics(idata_llm)
    rhat_df["n_divergences"] = n_div
    rhat_df["min_ess"] = min_ess
    rhat_df["outer_fold"] = outer_fold
    rhat_df["inner_fold"] = inner_fold
    rhat_df["eval_split"] = eval_split
    rhat_df["dimension"] = dimension
    rhat_df["judge_model"] = judge_model
    return rhat_df


def _store_array_once(artifact_arrays, key, value):
    if key not in artifact_arrays:
        artifact_arrays[key] = np.asarray(value)
    return key
