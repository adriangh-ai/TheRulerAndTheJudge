import numpy as np
import pandas as pd
import warnings

from itertools import combinations
from scipy.special import expit
from rich.table import Table
from rich import box

from scipy.stats import spearmanr, kendalltau


def _ensure_zero_based_labels(y_obs, n_cats):
    y_obs = np.asarray(y_obs)
    if y_obs.min() == 0 and y_obs.max() == n_cats - 1:
        return y_obs.astype(int)
    if y_obs.min() == 1 and y_obs.max() == n_cats:
        return (y_obs - 1).astype(int)
    raise ValueError(
        f"y_obs must be either 0..{n_cats-1} or 1..{n_cats}; "
        f"got min={y_obs.min()}, max={y_obs.max()}."
    )


def _probs_from_eta_draws(
    df,
    eta_draws_obs,          # shape (M, N)
    cuts_mean,
    y_obs,
    meta,
    alpha_mean=None,        # shape (n_raters,) if present
    beta_diff_mean=None,    # shape (n_raters, n_items) if present
):
    """
    Computes marginalized category probabilities, expected score, variance,
    and residual statistics for each observation under the Stage 2 model
    that propagates anchor uncertainty via posterior draws.

    Parameters
    ----------
    eta_draws_obs : np.ndarray, shape (M, N)
        Posterior draws of anchor eta = theta - beta, aligned to observations.
    cuts_mean : np.ndarray, shape (n_raters, K-1) or (1, K-1)
        Posterior mean cutpoints.
    y_obs : np.ndarray, shape (n_obs,)
        Observed ordinal labels, either 0..K-1 or 1..K.
    alpha_mean : np.ndarray or None
        Posterior mean alpha per rater if the alpha sensitivity model was fit.
    beta_diff_mean : np.ndarray or None
        Posterior mean DIF shift per (rater, item) if the DIF model was fit.

    Returns
    -------
    df_stats : pd.DataFrame
        Copy of df with columns:
        - E_y
        - var_y
        - resid_sq
        - z_sq
    fit_df : pd.DataFrame
        Infit / Outfit by rater.
    """
    df_stats = df.copy()

    n_obs = len(df_stats)
    n_cats = cuts_mean.shape[1] + 1
    #y_obs = _ensure_zero_based_labels(y_obs, n_cats=n_cats)

    r_idxs = df_stats["rater_idx"].values
    i_idxs = df_stats["item_idx"].values

    eta_draws_obs = np.asarray(eta_draws_obs, dtype=float)   # (M, N)
    if eta_draws_obs.ndim != 2 or eta_draws_obs.shape[1] != n_obs:
        raise ValueError(
            f"eta_draws_obs must have shape (M, {n_obs}), got {eta_draws_obs.shape}"
        )

    # Observation-specific cutpoints
    if cuts_mean.shape[0] == 1:
        row_cuts = np.repeat(cuts_mean, n_obs, axis=0)   # (N, K-1)
    else:
        row_cuts = cuts_mean[r_idxs]                     # (N, K-1)

    # Optional alpha per observation
    if alpha_mean is None:
        alpha_obs = np.ones(n_obs, dtype=float)
    else:
        alpha_obs = np.asarray(alpha_mean, dtype=float)[r_idxs]

    # Optional DIF shift per observation
    if beta_diff_mean is None:
        beta_diff_obs = np.zeros(n_obs, dtype=float)
    else:
        beta_diff_obs = np.asarray(beta_diff_mean, dtype=float)[r_idxs, i_idxs]

    # Apply optional transformations per observation
    eta_mn = eta_draws_obs - beta_diff_obs[None, :]
    eta_mn = alpha_obs[None, :] * eta_mn                  # (M, N)

    # Ordered-logistic probabilities at each posterior draw
    cdf = expit(row_cuts[None, :, :] - eta_mn[:, :, None])   # (M, N, K-1)

    left = np.concatenate(
        [np.zeros((cdf.shape[0], cdf.shape[1], 1)), cdf],
        axis=2
    )
    right = np.concatenate(
        [cdf, np.ones((cdf.shape[0], cdf.shape[1], 1))],
        axis=2
    )
    probs_m = right - left                                  # (M, N, K)
    probs_m = np.clip(probs_m, 1e-12, 1.0)
    probs_m = probs_m / probs_m.sum(axis=2, keepdims=True)

    # Marginalize over anchor posterior draws
    probs = probs_m.mean(axis=0)                            # (N, K)
    probs = np.clip(probs, 1e-12, 1.0)
    probs = probs / probs.sum(axis=1, keepdims=True)

    # Expected score and variance
    k_vals = np.arange(n_cats, dtype=float)
    E_y = np.sum(probs * k_vals[None, :], axis=1)
    E_y2 = np.sum(probs * (k_vals[None, :] ** 2), axis=1)
    Var_y = E_y2 - (E_y ** 2)
    Var_y = np.clip(Var_y, 1e-6, None)

    resid_sq = (y_obs - E_y) ** 2
    z_sq = resid_sq / Var_y

    df_stats["resid_sq"] = resid_sq
    df_stats["var_y"] = Var_y
    df_stats["z_sq"] = z_sq
    df_stats["E_y"] = E_y

    # Item level fit statistics:
    resid = y_obs - E_y
    z = resid / np.sqrt(Var_y)

    df_stats["y_obs"] = y_obs
    df_stats["resid"] = resid
    df_stats["z"] = z

    fit_stats = {}
    rater_names = [str(x) for x in meta["rater_categories"]]

    for ridx, r_name in enumerate(rater_names):
        subset = df_stats[df_stats["rater_idx"] == ridx]
        if len(subset) == 0:
            continue

        infit = np.sum(subset["resid_sq"]) / np.sum(subset["var_y"])
        outfit = np.mean(subset["z_sq"])

        fit_stats[r_name] = {
            "Infit_MNSQ": float(infit),
            "Outfit_MNSQ": float(outfit),
            "N": int(len(subset)),
        }

    fit_df = pd.DataFrame(fit_stats).T
    return df_stats, fit_df


def extract_eta_anchor_draws_obs(
    idata_anchor,
    df,
    meta_anchor,
    theta_var_name="theta",
    beta_var_name="beta",
    anchor_draw_grid=None,
):
    """
    Extract posterior draws of eta = theta - beta aligned to the observations in df.

    Returns
    -------
    eta_draws_obs : np.ndarray, shape (M, N)
        Posterior draws of eta for each observation.
    """
    post = idata_anchor.posterior

    anchor_systems = [str(x) for x in post.coords["system"].values]
    anchor_items = [str(x) for x in post.coords["items"].values]

    system_lookup = {sid: i for i, sid in enumerate(anchor_systems)}
    item_lookup = {iid: i for i, iid in enumerate(anchor_items)}

    theta_draws = np.asarray(post[theta_var_name]).reshape(-1, len(anchor_systems))
    beta_draws = np.asarray(post[beta_var_name]).reshape(-1, len(anchor_items))

    if anchor_draw_grid is not None:
        theta_draws = theta_draws[anchor_draw_grid]
        beta_draws = beta_draws[anchor_draw_grid]

    # Prefer actual IDs if present, fall back to indices
    if "system_id" in df.columns:
        sys_idx = [system_lookup[str(s)] for s in df["system_id"].values]
    else:
        sys_idx = df["system_idx"].values

    if "item_id" in df.columns:
        item_idx = [item_lookup[str(i)] for i in df["item_id"].values]
    else:
        item_idx = df["item_idx"].values

    eta_draws_obs = theta_draws[:, sys_idx] - beta_draws[:, item_idx]   # (M, N)
    return eta_draws_obs.astype(np.float32)


def _safe_corr(x, y, method="spearman"):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]

    if len(x) < 3 or np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return np.nan

    if method == "spearman":
        return float(spearmanr(x, y).correlation)
    if method == "kendall":
        return float(kendalltau(x, y, variant="b").statistic)
    if method == "pearson":
        return float(np.corrcoef(x, y)[0, 1])

    raise ValueError(method)


def calculate_stage2_within_system_item_alignment(df_stats, meta):
    """
    Within-system item-pattern alignment.

    For each judge and system, correlate observed judge scores over items
    with Stage-2 expected scores over items.

    This directly tests whether the judge assigns high scores to the items
    that the human anchor expects to be high for that system.
    """
    rows = []

    rater_names = [str(x) for x in meta["rater_categories"]]

    system_col = "system_id" if "system_id" in df_stats.columns else "system_idx"

    for (ridx, sid), sub in df_stats.groupby(["rater_idx", system_col]):
        if len(sub) < 3:
            continue

        obs = sub["y_obs"].to_numpy(float)
        exp = sub["E_y"].to_numpy(float)

        rows.append({
            "rater_idx": int(ridx),
            "rater_id": rater_names[int(ridx)],
            "system_id": str(sid),
            "N_items": int(len(sub)),
            "within_system_item_spearman": _safe_corr(obs, exp, "spearman"),
            "within_system_item_kendall": _safe_corr(obs, exp, "kendall"),
            "within_system_item_pearson": _safe_corr(obs, exp, "pearson"),
            "mean_abs_z": float(sub["z"].abs().mean()),
            "mean_z": float(sub["z"].mean()),
            "outfit_system": float(sub["z_sq"].mean()),
        })

    detail = pd.DataFrame(rows)

    if detail.empty:
        return detail, pd.DataFrame()

    summary = (
        detail
        .groupby(["rater_idx", "rater_id"])
        .agg(
            n_systems=("system_id", "nunique"),
            mean_within_system_item_spearman=("within_system_item_spearman", "mean"),
            median_within_system_item_spearman=("within_system_item_spearman", "median"),
            mean_within_system_item_kendall=("within_system_item_kendall", "mean"),
            mean_within_system_item_pearson=("within_system_item_pearson", "mean"),
            mean_abs_z=("mean_abs_z", "mean"),
            mean_system_outfit=("outfit_system", "mean"),
        )
        .reset_index()
    )

    return detail, summary


def extract_beta_means(idata_anchor, beta_var_name="beta"):
    post = idata_anchor.posterior
    item_names = [str(x) for x in post.coords["items"].values]
    beta_mean = np.asarray(post[beta_var_name].mean(dim=["chain", "draw"]).values)
    return pd.DataFrame({
        "item_id": item_names,
        "beta_mean": beta_mean.astype(float),
    })


def calculate_beta_residual_association(item_fit_df, beta_df):
    """
    For each judge, test whether item signed residuals correlate with human item difficulty.
    A strong association means the judge systematically disagrees with the human item ruler.
    """
    df = item_fit_df.merge(beta_df, on="item_id", how="left")
    rows = []

    for rater_id, sub in df.groupby("rater_id"):
        rows.append({
            "rater_id": rater_id,
            "n_items": int(sub["item_id"].nunique()),
            "spearman_beta_mean_z": _safe_corr(sub["beta_mean"], sub["mean_z"], "spearman"),
            "pearson_beta_mean_z": _safe_corr(sub["beta_mean"], sub["mean_z"], "pearson"),
            "spearman_beta_item_outfit": _safe_corr(sub["beta_mean"], sub["item_outfit_mnsq"], "spearman"),
            "mean_abs_item_mean_z": float(sub["mean_z"].abs().mean()),
            "max_abs_item_mean_z": float(sub["mean_z"].abs().max()),
        })

    return pd.DataFrame(rows)


def calculate_item_fit_step2(df_stats, meta):
    """
    Judge-specific item fit under the Stage-2 posterior-predictive model.

    This is not the same as Stage-1 human item fit. It asks:
    given the human anchor beta_i and judge cutpoints, does judge m score
    item i as expected across systems?
    """
    rows = []

    rater_names = [str(x) for x in meta["rater_categories"]]
    item_names = [str(x) for x in meta["item_categories"]]

    for (ridx, iidx), sub in df_stats.groupby(["rater_idx", "item_idx"]):
        if len(sub) == 0:
            continue

        infit = sub["resid_sq"].sum() / max(sub["var_y"].sum(), 1e-12)
        outfit = sub["z_sq"].mean()

        rows.append({
            "rater_idx": int(ridx),
            "rater_id": rater_names[int(ridx)],
            "item_idx": int(iidx),
            "item_id": item_names[int(iidx)],
            "N": int(len(sub)),
            "item_infit_mnsq": float(infit),
            "item_outfit_mnsq": float(outfit),
            "mean_resid": float(sub["resid"].mean()),
            "mean_z": float(sub["z"].mean()),
            "mean_abs_z": float(sub["z"].abs().mean()),
            "max_abs_z": float(sub["z"].abs().max()),
            "obs_mean": float(sub["y_obs"].mean()),
            "expected_mean": float(sub["E_y"].mean()),
        })

    return pd.DataFrame(rows)


def calculate_rasch_fit_statistics_step2(
    df,
    idata,
    meta,
    idata_anchor=None,
    anchor_draw_grid=None,
    random_state=42,
):
    """
    Calculates Infit and Outfit MNSQ for each rater under the Stage 2 model
    with anchor uncertainty propagated via posterior draws.

    Parameters
    ----------
    df : pd.DataFrame
        Stage 2 dataframe.
    idata : arviz.InferenceData
        Stage 2 posterior.
    meta : dict
        Stage 2 metadata.
    idata_anchor : arviz.InferenceData
        Stage 1 anchor posterior. Required for draw-based propagation.
    anchor_draw_grid : np.ndarray, optional
        Indices of Stage 1 posterior draws to use for fit analysis.
    """
    if idata_anchor is None:
        raise ValueError("idata_anchor is required for draw-based Stage 2 fit analysis.")

    post = idata.posterior

    # Posterior mean cutpoints from Stage 2
    cuts_mean = np.asarray(
        post["llm_cuts"].mean(dim=["chain", "draw"]).values
    )  # (n_raters, K-1)

    y_obs = df["y_ord"].values

    # Optional alpha model
    alpha_mean = None
    if "log_alpha" in post:
        alpha_mean = np.exp(
            np.asarray(post["log_alpha"].mean(dim=["chain", "draw"]).values)
        )
    elif "alpha" in post:
        alpha_mean = np.asarray(
            post["alpha"].mean(dim=["chain", "draw"]).values
        )

    # Optional DIF model
    beta_diff_mean = None
    if "beta_diff" in post:
        beta_diff_mean = np.asarray(
            post["beta_diff"].mean(dim=["chain", "draw"]).values
        )  # (n_raters, n_items)

    eta_draws_obs = extract_eta_anchor_draws_obs(
        idata_anchor=idata_anchor,
        df=df,
        meta_anchor=None,
        theta_var_name="theta",
        beta_var_name="beta",
        anchor_draw_grid=anchor_draw_grid,
    )

    df_stats, fit_df = _probs_from_eta_draws(
        df=df,
        eta_draws_obs=eta_draws_obs,
        cuts_mean=cuts_mean,
        y_obs=y_obs,
        meta=meta,
        alpha_mean=alpha_mean,
        beta_diff_mean=beta_diff_mean,
    )

    item_fit_df = calculate_item_fit_step2(df_stats, meta)

    alignment_detail_df, alignment_summary_df = calculate_stage2_within_system_item_alignment(
        df_stats,
        meta,
    )

    beta_resid_df = None
    if idata_anchor is not None:
        beta_df = extract_beta_means(idata_anchor)
        beta_resid_df = calculate_beta_residual_association(item_fit_df, beta_df)

    return {
        "observation_stats": df_stats,
        "rater_fit": fit_df,
        "item_fit": item_fit_df,
        "within_system_item_alignment_detail": alignment_detail_df,
        "within_system_item_alignment_summary": alignment_summary_df,
        "beta_residual_association": beta_resid_df,
    }


    return _probs_from_eta_draws(
        df=df,
        eta_draws_obs=eta_draws_obs,
        cuts_mean=cuts_mean,
        y_obs=y_obs,
        meta=meta,
        alpha_mean=alpha_mean,
        beta_diff_mean=beta_diff_mean,
    )


#-----------------------------------------------------------------------
# Rasch Fit Statistics Calculation and Visualization
#-----------------------------------------------------------------------

def _probs_from_eta(df, eta, cuts_mean, y_obs, meta):
    """
    Computes category probabilities, expected score, variance, and residual stats
    for each observation.

    Returns
    -------
    df_stats : pd.DataFrame
        Copy of df with extra columns:
        - E_y
        - Var_y
        - resid_sq
        - z_sq
    probs : np.ndarray
        Observation x category probability matrix
    """
    df_stats = df.copy()
    # We iterate 0..4 (or 0..K)
    n_obs = len(df_stats)
    n_cats = cuts_mean.shape[1] + 1
    
    probs = np.zeros((n_obs, n_cats))
    
    # Sigmoid function
    if cuts_mean.shape[0] == 1:
        # If cuts are the same for all raters, we can skip indexing
        cuts_mean = np.repeat(cuts_mean, n_obs, axis=0)
    for k in range(n_cats):
        # Cuts are for the rater of each row
        # Row-specific cuts: cuts_mean[r_idxs]
        row_cuts = cuts_mean[df_stats["rater_idx"].values]
        
        if k == 0:
            # P(y=0) = 1 - Sigmoid(eta - c0)
            p = 1.0 - expit(eta - row_cuts[:, 0])
        elif k == n_cats - 1:
            # P(y=last) = Sigmoid(eta - c_last)
            p = expit(eta - row_cuts[:, -1])
        else:
            # P(y=k) = Sig(eta - c_{k-1}) - Sig(eta - c_k)
            c_low = row_cuts[:, k-1]
            c_high = row_cuts[:, k]
            p = expit(eta - c_low) - expit(eta - c_high)
        probs[:, k] = p

    # 3. Calculate Expectations (First and Second Moment)
    k_vals = np.arange(n_cats)
    E_y = np.sum(probs * k_vals, axis=1) # Expected Score
    E_y2 = np.sum(probs * (k_vals**2), axis=1)
    Var_y = E_y2 - (E_y**2)
    
    # 4. Calculate Residuals (Standardized)
    # Z = (Observed - Expected) / sqrt(Variance)
    # Clamp variance to avoid division by zero
    Var_y = np.clip(Var_y, 1e-4, 10.0)
    
    resid_sq = (y_obs - E_y)**2
    z_sq = resid_sq / Var_y
    
    # 5. Aggregate by Rater (MNSQ)
    # Infit = Sum(Resid^2) / Sum(Var)  <-- Weighted by information
    # Outfit = Mean(Z^2)               <-- Unweighted (sensitive to outliers)
    
    df_stats["resid_sq"] = resid_sq
    df_stats["var_y"] = Var_y
    df_stats["z_sq"] = z_sq
    df_stats["E_y"] = E_y
    
    fit_stats = {}
    
    for r_name in meta["rater_categories"]:
        subset = df_stats[df_stats["rater_id"] == r_name]
        if len(subset) == 0: continue
            
        # Infit: Information-weighted mean square
        infit = np.sum(subset["resid_sq"]) / np.sum(subset["var_y"])
        
        # Outfit: Unweighted mean square
        outfit = np.mean(subset["z_sq"])
        
        fit_stats[r_name] = {"Infit_MNSQ": infit, "Outfit_MNSQ": outfit}
        
    return df_stats, pd.DataFrame(fit_stats).T


def _aggregate_mnsq(df_stats, group_col, label_map=None):
    """
    Aggregate observation-level residual stats into Infit/Outfit MNSQ
    for any grouping facet (e.g. raters, items).

    Parameters
    ----------
    df_stats : pd.DataFrame
        Must contain group_col, resid_sq, Var_y, z_sq
    group_col : str
        Column to group by, e.g. 'rater_idx', 'item_idx'
    label_map : dict or None
        Optional mapping from raw group values to display labels

    Returns
    -------
    pd.DataFrame indexed by group label with columns:
        - Infit_MNSQ
        - Outfit_MNSQ
        - N
    """
    fit_stats = {}

    for g, subset in df_stats.groupby(group_col, sort=True):
        if len(subset) == 0:
            continue

        infit = np.sum(subset["resid_sq"]) / np.sum(subset["var_y"])
        outfit = np.mean(subset["z_sq"])

        label = label_map[g] if label_map is not None else g
        fit_stats[label] = {
            "Infit_MNSQ": infit,
            "Outfit_MNSQ": outfit,
        }

    return pd.DataFrame(fit_stats).T.sort_values("Outfit_MNSQ", ascending=False)


def calculate_rasch_fit_statistics_human_items(df, idata, meta):
    """
    Calculates item-level Infit and Outfit MNSQ for the human Rasch/MFRM model.

    Interpretation:
    - High item underfit (>1.5): item may be noisy, ambiguous, multidimensional,
      or otherwise not behaving like the same construct as the rest.
    - Low item overfit (<0.5): item may be overly predictable / redundant.
    """
    post = idata.posterior

    # Posterior means
    rho_mean = post["rho_human"].mean(dim=["chain", "draw"]).values
    theta_means = post["theta"].mean(dim=["chain", "draw"]).values
    beta_means = post["beta"].mean(dim=["chain", "draw"]).values
    cutpoints = post["cutpoints"].mean(dim=["chain", "draw"]).values

    # Expected shape: (n_raters, n_cutpoints)
    cuts_mean = np.asarray(cutpoints)
    if cuts_mean.ndim == 1:
        cuts_mean = cuts_mean[None, :]

    # Indices
    r_idxs = df["rater_idx"].values
    s_idxs = df["system_idx"].values
    i_idxs = df["item_idx"].values
    y_obs = df["y_ord"].values

    # Human eta
    theta_n = theta_means[s_idxs]
    beta_i = beta_means[i_idxs]
    rho_j = rho_mean[r_idxs]
    eta = theta_n - beta_i - rho_j

    # Observation-level stats
    df_stats, rater_fit = _probs_from_eta(df, eta, cuts_mean, y_obs, meta)

    # Optional prettier labels
    item_labels = None
    if "item_categories" in meta:
        item_labels = {i: name for i, name in enumerate(meta["item_categories"])}

    fit_df = _aggregate_mnsq(
        df_stats=df_stats,
        group_col="item_idx",
        label_map=item_labels,
    )

    return fit_df, rater_fit


def get_rasch_mnsq_style(val):
    """
    Returns style based on Linacre's Guidelines for Rating Scales:
    > 1.5: Unproductive construction of measurement (Noisy) -> Red
    < 0.5: Less productive, but not degrading (Rigid/Overfit) -> Yellow
    0.5 - 1.5: Productive for measurement -> Green
    """
    if val > 1.5:
        return "bold red" # Degrading/Noisy
    elif val < 0.5:
        return "bold yellow" # Muted/Rigid (often acceptable for LLMs)
    else:
        return "bold green" # Ideal


def print_rasch_fit_table(fit_df, console, type="LLM", dim="Overall"):
    """
    Visualizes the Rasch Fit Statistics DataFrame in Rich style.
    """
    # Create Table
    table = Table(
        title=f"Rasch Fit Statistics (MNSQ) - {type} Raters - Dimension: {dim}",
        caption="Rule of Thumb: Ideal [0.5, 1.5] | >1.5 (Noisy) | <0.5 (Rigid)",
        box=box.ROUNDED,
        header_style="bold cyan"
    )

    # Add Columns
    table.add_column("Rater ID", style="bold white")
    table.add_column("Infit MNSQ", justify="right")
    table.add_column("Outfit MNSQ", justify="right")
    table.add_column("Diagnosis", justify="center", style="italic")

    # Sort by Outfit MNSQ descending (Noisiest first)
    sorted_df = fit_df.sort_values("Outfit_MNSQ", ascending=False)

    for rater_id, row in sorted_df.iterrows():
        infit = row["Infit_MNSQ"]
        outfit = row["Outfit_MNSQ"]

        # Determine Styles
        infit_style = get_rasch_mnsq_style(infit)
        outfit_style = get_rasch_mnsq_style(outfit)
        
        # Simple diagnosis text
        if outfit > 1.5:
            diag = "[red]Noisy/Random[/]"
        elif outfit < 0.5:
            diag = "[yellow]Rigid/Redundant[/]"
        else:
            diag = "[green]Productive[/]"

        table.add_row(
            str(rater_id),
            f"[{infit_style}]{infit:.2f}[/]",
            f"[{outfit_style}]{outfit:.2f}[/]",
            diag
        )

    console.print(table)


# -------------------------------------------------------------------------
# CSEM and Information Curves for Human Rasch Model
# -------------------------------------------------------------------------

def ordered_probs_exp_var_info(eta, cutpoints):
    """
    eta: shape (...,)
    cutpoints: shape (K-1,)
    returns:
      probs (..., K)
      expected_score (...,)
      var_score (...,)
      fisher_info (...,) wrt theta (since d eta / d theta = 1)

    Scores are on 1..K scale.
    """
    eta = np.asarray(eta, dtype=np.float64)
    tau = np.asarray(cutpoints, dtype=np.float64).squeeze()
    K = tau.size + 1

    cuts_ext = np.concatenate(([-np.inf], tau, [np.inf]))
    probs = np.empty(eta.shape + (K,), dtype=np.float64)
    dprobs = np.empty_like(probs)

    for k in range(K):
        lo = cuts_ext[k]
        hi = cuts_ext[k + 1]

        if np.isneginf(lo):
            s_lo = np.ones_like(eta)
            ds_lo = np.zeros_like(eta)
        else:
            s_lo = expit(eta - lo)
            ds_lo = s_lo * (1.0 - s_lo)

        if np.isposinf(hi):
            s_hi = np.zeros_like(eta)
            ds_hi = np.zeros_like(eta)
        else:
            s_hi = expit(eta - hi)
            ds_hi = s_hi * (1.0 - s_hi)

        pk = s_lo - s_hi
        dpk = ds_lo - ds_hi

        probs[..., k] = np.clip(pk, 1e-12, 1.0)
        dprobs[..., k] = dpk

    probs = probs / probs.sum(axis=-1, keepdims=True)

    cats = np.arange(1, K + 1, dtype=np.float64)
    exp_score = np.sum(probs * cats, axis=-1)
    var_score = np.sum(probs * (cats - exp_score[..., None]) ** 2, axis=-1)
    fisher_info = np.sum((dprobs ** 2) / np.clip(probs, 1e-12, None), axis=-1)

    return probs, exp_score, var_score, fisher_info


def _coerce_cutpoint_matrix(cutpoints_mean, n_raters):
    """
    Ensures cutpoints are (n_raters, K-1).
    Handles either:
      - common cutpoints: (K-1,)
      - rater-specific cutpoints: (n_raters, K-1)
    """
    cuts = np.asarray(cutpoints_mean, dtype=np.float64)

    if cuts.ndim == 1:
        cuts = np.tile(cuts[None, :], (n_raters, 1))
    elif cuts.ndim == 2:
        if cuts.shape[0] != n_raters:
            raise ValueError(
                f"cutpoints first dimension ({cuts.shape[0]}) != n_raters ({n_raters})"
            )
    else:
        raise ValueError(f"Unexpected cutpoints shape: {cuts.shape}")

    return cuts


def supported_resolution_from_draws(
    idata,
    tau_h: float = 0.8,
    theta_var_name: str = "theta",
):
    """
    Compute Stage-1 Supported Resolution (SR) from full human posterior draws.

    Parameters
    ----------
    idata : arviz.InferenceData
        Must contain posterior draws for theta with dims (chain, draw, system).
    tau_h : float
        Human dominance-certainty threshold in [0,1].
    theta_var_name : str
        Name of the theta variable in idata.posterior.

    Returns
    -------
    result : dict
        {
            "sr": float,
            "n_systems": int,
            "n_pairs": int,
            "tau_h": float,
            "pairwise_p": dict[(a,b)] -> p_ab,
            "pairwise_omega": dict[(a,b)] -> omega_ab,
            "supported_pairs": list[(a,b)]
        }
    """
    if not (0.0 <= tau_h <= 1.0):
        raise ValueError("tau_h must be in [0,1].")

    theta = idata.posterior[theta_var_name].values  # expected shape: (chain, draw, system)
    if theta.ndim != 3:
        raise ValueError(
            f"Expected posterior[{theta_var_name}] to have shape (chain, draw, system), "
            f"got shape {theta.shape}"
        )

    n_chains, n_draws, n_systems = theta.shape
    theta_flat = theta.reshape(n_chains * n_draws, n_systems)  # (T, S)

    pairwise_p = {}
    pairwise_omega = {}
    supported_pairs = []

    for a, b in combinations(range(n_systems), 2):
        p_ab = float(np.mean(theta_flat[:, a] > theta_flat[:, b]))
        omega_ab = abs(2.0 * p_ab - 1.0)

        pairwise_p[(a, b)] = p_ab
        pairwise_omega[(a, b)] = omega_ab

        if omega_ab >= tau_h:
            supported_pairs.append((a, b))

    n_pairs = n_systems * (n_systems - 1) // 2
    sr = float(len(supported_pairs) / n_pairs) if n_pairs > 0 else np.nan

    return {
        "sr": sr,
        "n_systems": n_systems,
        "n_pairs": n_pairs,
        "tau_h": tau_h,
        "pairwise_p": pairwise_p,
        "pairwise_omega": pairwise_omega,
        "supported_pairs": supported_pairs,
    }


def benchmark_information_curve_human(
    df,
    idata,
    meta,
    theta_grid=None,
    average_per_system=True,
):
    """
    Computes benchmark information and CSEM for the human Stage-1 model.

    IMPORTANT:
    - Information is aggregated over the empirical human design.
    - If average_per_system=True, we divide by the number of systems so that
      the result corresponds to a typical evaluated system, not the full dataset.

    Returns
    -------
    curve_df : pd.DataFrame
        Columns: theta, information, csem
    theta_hat : np.ndarray
        Posterior mean human theta per system
    theta_sd : np.ndarray
        Posterior SD human theta per system
    """
    post = idata.posterior

    theta_hat = post["theta"].mean(dim=["chain", "draw"]).values.astype(np.float64)
    theta_sd = post["theta"].std(dim=["chain", "draw"]).values.astype(np.float64)
    beta_hat = post["beta"].mean(dim=["chain", "draw"]).values.astype(np.float64)
    rho_hat = post["rho_human"].mean(dim=["chain", "draw"]).values.astype(np.float64)
    cutpoints_mean = post["cutpoints"].mean(dim=["chain", "draw"]).values

    n_raters = len(meta["rater_categories"])
    cuts = _coerce_cutpoint_matrix(cutpoints_mean, n_raters)

    if theta_grid is None:
        warnings.warn("theta_grid is None; using default grid", UserWarning)
        lo = float(np.floor(theta_hat.min() - 1.0))
        hi = float(np.ceil(theta_hat.max() + 1.0))
        theta_grid = np.linspace(lo, hi, 241)

    theta_grid = np.asarray(theta_grid, dtype=np.float64)

    item_idx = df["item_idx"].to_numpy()
    rater_idx = df["rater_idx"].to_numpy()
    system_idx = df["system_idx"].to_numpy()

    info_total = np.zeros_like(theta_grid, dtype=np.float64)

    # Group by rater because cutpoints may be rater-specific
    for r in np.unique(rater_idx):
        mask = (rater_idx == r)
        if not np.any(mask):
            continue

        beta_r = beta_hat[item_idx[mask]]              # (#obs_for_rater,)
        rho_r = float(rho_hat[r])                      # scalar
        cuts_r = cuts[r]                               # (K-1,)

        # eta(theta) for every observation attached to this rater:
        # shape = (N_theta, N_obs_r)
        eta = theta_grid[:, None] - beta_r[None, :] - rho_r

        _, _, _, fisher_info = ordered_probs_exp_var_info(eta, cuts_r)
        # sum across the observation slots of the empirical design
        info_total += fisher_info.sum(axis=1)

    if average_per_system:
        n_systems = max(int(np.unique(system_idx).size), 1)
        info_total = info_total / n_systems

    csem = 1.0 / np.sqrt(np.clip(info_total, 1e-12, None))

    curve_df = pd.DataFrame({
        "theta": theta_grid,
        "information": info_total,
        "csem": csem,
    })

    return curve_df, theta_hat, theta_sd


def classify_resolution_ratio(ratio):
    """
    These are proposed decision thresholds for THIS paper.
    They are not canonical Rasch cutoffs.
    """
    if ratio <= 0.25:
        return "fine-grained"
    elif ratio <= 0.50:
        return "coarse but usable"
    elif ratio <= 1.00:
        return "blunt"
    else:
        return "unacceptable"


def summarize_benchmark_resolution(
        df,
        idata,
        meta,
        theta_grid=None,
        central_mass=0.80,
        z_value=1.96,
        average_per_system=True,
        sr_tau_h=0.8,
    ):
    curve_df, theta_hat, theta_sd = benchmark_information_curve_human(
        df=df,
        idata=idata,
        meta=meta,
        theta_grid=theta_grid,
        average_per_system=average_per_system,
    )

    alpha_tail = (1.0 - central_mass) / 2.0
    q_lo = float(np.quantile(theta_hat, alpha_tail))
    q_hi = float(np.quantile(theta_hat, 1.0 - alpha_tail))
    spread80 = float(q_hi - q_lo)

    occupied = curve_df[(curve_df["theta"] >= q_lo) & (curve_df["theta"] <= q_hi)].copy()
    if occupied.empty:
        nearest_idx = np.argmin(np.abs(curve_df["theta"].to_numpy() - np.median(theta_hat)))
        occupied = curve_df.iloc[[nearest_idx]].copy()

    worst_idx = occupied["csem"].idxmax()
    csem80 = float(occupied.loc[worst_idx, "csem"])
    theta_worst = float(occupied.loc[worst_idx, "theta"])
    info80_min = float(occupied.loc[worst_idx, "information"])

    delta95 = float(z_value * np.sqrt(2.0) * csem80)

    ratio = float(delta95 / spread80) if spread80 > 1e-12 else np.inf
    label = classify_resolution_ratio(ratio)

    observed_var = float(np.var(theta_hat, ddof=1)) if len(theta_hat) > 1 else np.nan
    error_var = float(np.mean(theta_sd ** 2))
    if np.isfinite(observed_var) and observed_var > 0:
        reliability = max(0.0, min(1.0, (observed_var - error_var) / observed_var))
    else:
        reliability = np.nan

    sr_result = supported_resolution_from_draws(
        idata=idata,
        tau_h=sr_tau_h,
        theta_var_name="theta",
    )

    summary = {
        "theta_q10": q_lo,
        "theta_q90": q_hi,
        "spread80": spread80,
        "csem80": csem80,
        "theta_worst_csem": theta_worst,
        "information_min_in_80": info80_min,
        "delta95": delta95,
        "resolution_ratio": ratio,
        "resolution_label": label,
        "theta_observed_var": observed_var,
        "theta_error_var": error_var,
        "theta_reliability": reliability,
        "supported_resolution": sr_result["sr"],
        "supported_resolution_tau_h": sr_tau_h,
        "n_supported_pairs": len(sr_result["supported_pairs"]),
        "n_pairs": sr_result["n_pairs"],
    }

    return summary, curve_df, sr_result


def print_benchmark_resolution_card(summary, console, dim="Overall"):
    """
    Pretty printer for the actionable benchmark card.
    """
    table = Table(
        title=f"Benchmark Resolution Card - Dimension: {dim}",
        box=box.ROUNDED,
        header_style="bold cyan"
    )

    table.add_column("Metric", style="bold white")
    table.add_column("Value", justify="right")
    table.add_column("Interpretation", style="italic")

    label = summary["resolution_label"]
    if label == "fine-grained":
        style = "bold green"
    elif label == "coarse but usable":
        style = "bold yellow"
    elif label == "blunt":
        style = "bold red"
    else:
        style = "bold red"

    interp = (
        f"[{style}]{label}[/]: "
        f"Δ95 / spread80 = {summary['resolution_ratio']:.2f}"
    )

    table.add_row("θ q10", f"{summary['theta_q10']:.3f}", "lower occupied bound")
    table.add_row("θ q90", f"{summary['theta_q90']:.3f}", "upper occupied bound")
    table.add_row("Spread80", f"{summary['spread80']:.3f}", "central 80% spread of systems")
    table.add_row("CSEM^80", f"{summary['csem80']:.3f}", "worst CSEM in occupied region")
    table.add_row("Δ95", f"{summary['delta95']:.3f}", "approx. 95% pairwise resolution threshold")
    table.add_row("Worst θ", f"{summary['theta_worst_csem']:.3f}", "where occupied-region CSEM peaks")
    table.add_row("Decision", "", interp)
    table.add_row(
        f"SR (τ_h={summary['supported_resolution_tau_h']})", 
        f"{summary['supported_resolution']:.2%}",
        "Supported Resolution (SR)"
    )
    table.add_row(
        "SR Pairs", 
        f"{summary['n_supported_pairs']} / {summary['n_pairs']}", 
        "Number of pairwise comparisons with human dominance-certainty above τ_h"
    )

    console.print(table)

