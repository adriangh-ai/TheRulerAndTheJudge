import numpy as np
from scipy.special import logsumexp, expit


def compute_ordered_logprob_matrix(eta_matrix, cutpoints, k):
    """
    Matrix version of ordered logit log P(y=k | eta, cutpoints).
    eta_matrix: (N_theta, N_nodes)  (or any broadcastable 2D)
    cutpoints:  (K-1,)
    k:          integer category in 1..K (NOT 0-index)
    """
    # extend cutpoints with -inf/+inf
    cuts_ext = np.concatenate([[-np.inf], np.asarray(cutpoints), [np.inf]])
    c_lower = cuts_ext[k-1]
    c_upper = cuts_ext[k]

    # P(y >= k) = sigmoid(eta - c_{k-1})
    # P(y >= k+1) = sigmoid(eta - c_k)
    if np.isneginf(c_lower):
        p_ge_lower = np.ones_like(eta_matrix)
    else:
        p_ge_lower = expit(eta_matrix - c_lower)

    if np.isposinf(c_upper):
        p_ge_upper = np.zeros_like(eta_matrix)
    else:
        p_ge_upper = expit(eta_matrix - c_upper)

    prob = p_ge_lower - p_ge_upper
    return np.log(np.maximum(prob, 1e-15))




def get_system_posterior_human(
    group_df_human,
    human_rater_params,
    human_cutpoints,
    item_difficulty_map,
    theta_grid=np.linspace(-6, 6, 200)
):
    """
    Posterior over theta for ONE system using HUMAN scores (as 'truth proxy').
    group_df_human: rows for a single system with columns item_id, rater_id, score
    human_rater_params[rater_id] must contain rho (rater bias) OR set rho=0.
    human_cutpoints: (K-1,) on the same ordinal mapping
    """
    theta_grid = np.asarray(theta_grid)
    log_prior = np.zeros_like(theta_grid)
    total_log_like = np.zeros_like(theta_grid)

    for _, row in group_df_human.iterrows():
        item_id = row["item_id"]
        rater_id = row["rater_id"]
        score = int(row["score"])

        beta_center = item_difficulty_map.get(item_id, 0.0)  # uses same map; OK for fixed benchmark items
        rho = float(human_rater_params.get(rater_id, 0.0))

        eta = theta_grid - beta_center - rho
        # compute log prob for 1D eta by making it (N_theta, 1)
        logp = compute_ordered_logprob_matrix(eta[:, None], human_cutpoints, score).squeeze(1)
        total_log_like += logp

    unnormalized_log_post = total_log_like + log_prior
    log_evidence = logsumexp(unnormalized_log_post)
    posterior = np.exp(unnormalized_log_post - log_evidence)
    return posterior


def ordered_probs_from_eta(eta, cutpoints):
    """
    eta: array (...,)
    returns probs (..., K)
    """
    cutpoints = np.asarray(cutpoints)
    K = cutpoints.shape[0] + 1
    cuts_ext = np.concatenate([[-np.inf], cutpoints, [np.inf]])

    probs = []
    for k in range(1, K+1):
        c_lower = cuts_ext[k-1]
        c_upper = cuts_ext[k]

        if np.isneginf(c_lower):
            p_ge_lower = np.ones_like(eta)
        else:
            p_ge_lower = expit(eta - c_lower)

        if np.isposinf(c_upper):
            p_ge_upper = np.zeros_like(eta)
        else:
            p_ge_upper = expit(eta - c_upper)

        probs.append(p_ge_lower - p_ge_upper)

    P = np.stack(probs, axis=-1)
    P = np.clip(P, 1e-15, 1.0)
    P = P / P.sum(axis=-1, keepdims=True)
    return P


def predict_human_item_distribution(
    theta_grid,
    theta_posterior,
    beta_i,
    human_cutpoints,
    human_rho_values=None
):
    """
    Returns p(y=1..K | LLM data) for one (system,item) by integrating over theta posterior
    and (optionally) mixing over human rater bias distribution.
    """
    theta_grid = np.asarray(theta_grid)
    theta_posterior = np.asarray(theta_posterior)
    theta_posterior = theta_posterior / theta_posterior.sum()

    if human_rho_values is None or len(human_rho_values) == 0:
        eta = theta_grid - beta_i
        P_k_given_theta = ordered_probs_from_eta(eta, human_cutpoints)  # (N_theta, K)
        return (theta_posterior[:, None] * P_k_given_theta).sum(axis=0)

    # Mixture over human raters (random rater)
    human_rho_values = np.asarray(human_rho_values, dtype=np.float64)
    # P(y|theta) = average_rater P(y|theta, rho_r)
    P_acc = None
    for rho in human_rho_values:
        eta = theta_grid - beta_i - rho
        P = ordered_probs_from_eta(eta, human_cutpoints)  # (N_theta, K)
        if P_acc is None:
            P_acc = P
        else:
            P_acc += P
    P_acc /= len(human_rho_values)

    return (theta_posterior[:, None] * P_acc).sum(axis=0)


# Diagnosing Theta posterior sample grid quantile 
def posterior_mean_from_grid(posterior: np.ndarray, theta_grid: np.ndarray) -> float:
    posterior = np.asarray(posterior, dtype=np.float64)
    theta_grid = np.asarray(theta_grid, dtype=np.float64)

    posterior = posterior / posterior.sum()
    return float(np.sum(posterior * theta_grid))


def posterior_var_from_grid(posterior: np.ndarray, theta_grid: np.ndarray) -> float:
    posterior = np.asarray(posterior, dtype=np.float64)
    theta_grid = np.asarray(theta_grid, dtype=np.float64)

    posterior = posterior / posterior.sum()
    mean = np.sum(posterior * theta_grid)
    var = np.sum(posterior * (theta_grid - mean) ** 2)
    return float(var)


def posterior_quantile_from_grid(
    posterior: np.ndarray,
    theta_grid: np.ndarray,
    q: float,
) -> float:
    """
    Compute an approximate posterior quantile from a discrete grid posterior.

    Parameters
    ----------
    posterior : np.ndarray
        Posterior mass vector over theta_grid.
    theta_grid : np.ndarray
        Shared latent grid, sorted ascending.
    q : float
        Quantile in [0, 1].

    Returns
    -------
    float
        Approximate posterior quantile.
    """
    if not (0.0 <= q <= 1.0):
        raise ValueError("q must be in [0, 1].")

    posterior = np.asarray(posterior, dtype=np.float64)
    theta_grid = np.asarray(theta_grid, dtype=np.float64)

    if posterior.shape != theta_grid.shape:
        raise ValueError("posterior and theta_grid must have the same shape.")

    posterior = posterior / posterior.sum()
    cdf = np.cumsum(posterior)
    idx = np.searchsorted(cdf, q, side="left")
    idx = min(idx, len(theta_grid) - 1)
    return float(theta_grid[idx])


def boundary_mass(
    posterior: np.ndarray,
    theta_grid: np.ndarray,
    edge_bins: int = 3,
) -> dict:
    """
    Compute posterior mass near the left and right boundaries of the grid.

    Parameters
    ----------
    posterior : np.ndarray
        Posterior mass vector.
    theta_grid : np.ndarray
        Shared latent grid.
    edge_bins : int
        Number of bins at each edge to count as boundary region.

    Returns
    -------
    dict
        {
            "left_mass": ...,
            "right_mass": ...,
            "total_boundary_mass": ...,
            "left_edge_theta_max": ...,
            "right_edge_theta_min": ...
        }
    """
    posterior = np.asarray(posterior, dtype=np.float64)
    theta_grid = np.asarray(theta_grid, dtype=np.float64)

    if posterior.shape != theta_grid.shape:
        raise ValueError("posterior and theta_grid must have the same shape.")
    if edge_bins < 1 or edge_bins > len(theta_grid) // 2:
        raise ValueError("edge_bins must be >=1 and <= len(theta_grid)//2.")

    posterior = posterior / posterior.sum()

    left_mass = float(np.sum(posterior[:edge_bins]))
    right_mass = float(np.sum(posterior[-edge_bins:]))

    return {
        "left_mass": left_mass,
        "right_mass": right_mass,
        "total_boundary_mass": left_mass + right_mass,
        "left_edge_theta_max": float(theta_grid[edge_bins - 1]),
        "right_edge_theta_min": float(theta_grid[-edge_bins]),
    }


def grid_coverage_diagnostics(
    posterior: np.ndarray,
    theta_grid: np.ndarray,
    edge_bins: int = 3,
    credible_mass: float = 0.99,
) -> dict:
    """
    Check whether a posterior is well-contained inside the grid.

    Diagnostics:
    - posterior mean and sd
    - lower/upper credible interval
    - whether the credible interval touches the grid boundary
    - boundary mass in outer bins

    Parameters
    ----------
    posterior : np.ndarray
        Posterior mass vector.
    theta_grid : np.ndarray
        Shared latent grid.
    edge_bins : int
        Number of edge bins used for boundary mass.
    credible_mass : float
        Credible mass to inspect, e.g. 0.95 or 0.99.

    Returns
    -------
    dict
        Diagnostic summary.
    """
    if not (0.0 < credible_mass < 1.0):
        raise ValueError("credible_mass must be in (0, 1).")

    posterior = np.asarray(posterior, dtype=np.float64)
    theta_grid = np.asarray(theta_grid, dtype=np.float64)
    posterior = posterior / posterior.sum()

    mean = posterior_mean_from_grid(posterior, theta_grid)
    var = posterior_var_from_grid(posterior, theta_grid)
    sd = np.sqrt(var)

    alpha = 1.0 - credible_mass
    q_low = posterior_quantile_from_grid(posterior, theta_grid, alpha / 2.0)
    q_high = posterior_quantile_from_grid(posterior, theta_grid, 1.0 - alpha / 2.0)

    bmass = boundary_mass(posterior, theta_grid, edge_bins=edge_bins)

    touches_left = np.isclose(q_low, theta_grid[0]) or q_low <= theta_grid[0]
    touches_right = np.isclose(q_high, theta_grid[-1]) or q_high >= theta_grid[-1]

    return {
        "mean": mean,
        "sd": float(sd),
        f"ci_{int(credible_mass*100)}_low": q_low,
        f"ci_{int(credible_mass*100)}_high": q_high,
        "touches_left_boundary": bool(touches_left),
        "touches_right_boundary": bool(touches_right),
        **bmass,
    }


def assess_grid_width_for_systems(
    posteriors_by_system: dict,
    theta_grid: np.ndarray,
    edge_bins: int = 3,
    credible_mass: float = 0.99,
    boundary_mass_threshold: float = 1e-3,
) -> dict:
    """
    Assess whether the current grid width is adequate across many systems.

    Parameters
    ----------
    posteriors_by_system : dict
        Mapping {system_id: posterior_mass_vector}.
    theta_grid : np.ndarray
        Shared latent grid.
    edge_bins : int
        Number of edge bins used for boundary checks.
    credible_mass : float
        Credible interval mass for containment check.
    boundary_mass_threshold : float
        Threshold above which boundary mass is flagged.

    Returns
    -------
    dict
        Summary and per-system diagnostics.
    """
    per_system = {}
    flagged_boundary_mass = []
    flagged_interval_touch = []

    for system_id, posterior in posteriors_by_system.items():
        diag = grid_coverage_diagnostics(
            posterior,
            theta_grid,
            edge_bins=edge_bins,
            credible_mass=credible_mass,
        )
        per_system[system_id] = diag

        if diag["total_boundary_mass"] > boundary_mass_threshold:
            flagged_boundary_mass.append(system_id)

        if diag["touches_left_boundary"] or diag["touches_right_boundary"]:
            flagged_interval_touch.append(system_id)

    return {
        "theta_min": float(theta_grid[0]),
        "theta_max": float(theta_grid[-1]),
        "n_systems": len(posteriors_by_system),
        "boundary_mass_threshold": boundary_mass_threshold,
        "flagged_boundary_mass_systems": flagged_boundary_mass,
        "flagged_interval_touch_systems": flagged_interval_touch,
        "per_system": per_system,
    }


def compare_posteriors_across_grids(
    group_dfs_by_system: dict,
    judge_params,
    cutpoints,
    item_difficulty_map,
    get_system_posterior_fn,
    grid_small: np.ndarray,
    grid_large: np.ndarray,
    log_prior_small=None,
    log_prior_large=None,
) -> dict:
    """
    Recompute posteriors on two grids and compare posterior means and SDs.

    Parameters
    ----------
    group_dfs_by_system : dict
        Mapping {system_id: group_df}, where each group_df is the input expected
        by get_system_posterior_fn.
    judge_params, cutpoints, item_difficulty_map :
        Passed through to get_system_posterior_fn.
    get_system_posterior_fn : callable
        Function with the same interface as your get_system_posterior.
    grid_small : np.ndarray
        Current grid, e.g. np.linspace(-6, 6, 200)
    grid_large : np.ndarray
        Wider grid, e.g. np.linspace(-8, 8, 300)
    log_prior_small, log_prior_large :
        Optional priors for each grid.

    Returns
    -------
    dict
        Per-system and summary stability diagnostics.
    """
    results = {}
    delta_means = []
    delta_sds = []

    for system_id, group_df in group_dfs_by_system.items():
        post_small = get_system_posterior_fn(
            group_df=group_df,
            judge_params=judge_params,
            cutpoints=cutpoints,
            item_difficulty_map=item_difficulty_map,
            theta_grid=grid_small,
            log_prior=log_prior_small,
        )
        post_large = get_system_posterior_fn(
            group_df=group_df,
            judge_params=judge_params,
            cutpoints=cutpoints,
            item_difficulty_map=item_difficulty_map,
            theta_grid=grid_large,
            log_prior=log_prior_large,
        )

        mean_small = posterior_mean_from_grid(post_small, grid_small)
        mean_large = posterior_mean_from_grid(post_large, grid_large)

        sd_small = np.sqrt(posterior_var_from_grid(post_small, grid_small))
        sd_large = np.sqrt(posterior_var_from_grid(post_large, grid_large))

        d_mean = mean_large - mean_small
        d_sd = sd_large - sd_small

        delta_means.append(abs(d_mean))
        delta_sds.append(abs(d_sd))

        results[system_id] = {
            "mean_small": mean_small,
            "mean_large": mean_large,
            "delta_mean": d_mean,
            "sd_small": float(sd_small),
            "sd_large": float(sd_large),
            "delta_sd": d_sd,
        }

    return {
        "grid_small_range": (float(grid_small[0]), float(grid_small[-1])),
        "grid_large_range": (float(grid_large[0]), float(grid_large[-1])),
        "max_abs_delta_mean": float(np.max(delta_means)) if delta_means else np.nan,
        "mean_abs_delta_mean": float(np.mean(delta_means)) if delta_means else np.nan,
        "max_abs_delta_sd": float(np.max(delta_sds)) if delta_sds else np.nan,
        "mean_abs_delta_sd": float(np.mean(delta_sds)) if delta_sds else np.nan,
        "per_system": results,
    }

def row_to_soft_observation(row, n_classes=5, weight_mode="one", ess_value=None):
    """
    Convert one row with either:
      - score_list: list of repeated scores in 1..K
      - score: single hard score in 1..K
    into:
      q: (K,) probability vector
      w: observation weight
    """
    score_list = row.get("score_list", None)

    # score_list case
    if score_list is not None and score_list is not np.nan and len(score_list) > 0:
        arr = np.asarray(score_list, dtype=float).ravel()
        arr = arr[~np.isnan(arr)]  # remove NaNs
        arr = np.asarray(arr, dtype=int)

        if arr.min() < 1 or arr.max() > n_classes:
            raise ValueError(
                f"score_list values out of range [1,{n_classes}]: {arr}"
            )

        counts = np.bincount(arr, minlength=n_classes + 1)[1:]  # drop index 0
        R = counts.sum()
        q = counts / R

        if weight_mode == "count":
            w = float(R)          # exact equivalence to repeated hard labels
        elif weight_mode == "one":
            w = 1.0               # each item contributes once
        elif weight_mode == "ess":
            if ess_value is None:
                raise ValueError("ess_value must be provided for weight_mode='ess'")
            w = float(min(R, ess_value))
        else:
            raise ValueError(f"Unknown weight_mode={weight_mode}")

        return q, w

    # fallback hard-label case
    score = int(row["score"])
    if score < 1 or score > n_classes:
        raise ValueError(f"score out of range [1,{n_classes}]: {score}")

    q = np.zeros(n_classes, dtype=np.float64)
    q[score - 1] = 1.0
    w = 1.0
    return q, w


def compute_ordered_logprob_allcats(eta, cutpoints):
    """
    eta: (N_theta,)
    cutpoints: (K-1,)
    returns log_probs: (N_theta, K)
    """
    eta = np.asarray(eta, dtype=np.float64)
    cuts = np.asarray(cutpoints, dtype=np.float64)

    cuts_ext = np.concatenate([[-np.inf], cuts, [np.inf]])
    K = len(cuts_ext) - 1

    eta2 = eta[:, None]
    lower = cuts_ext[:-1][None, :]
    upper = cuts_ext[1:][None, :]

    p_ge_lower = np.where(np.isneginf(lower), 1.0, expit(eta2 - lower))
    p_ge_upper = np.where(np.isposinf(upper), 0.0, expit(eta2 - upper))

    probs = p_ge_lower - p_ge_upper
    probs = np.clip(probs, 1e-15, 1.0)
    probs = probs / probs.sum(axis=1, keepdims=True)

    return np.log(probs)


def get_system_posterior(
    group_df,
    judge_params,
    cutpoints,
    item_difficulty_map,
    theta_grid=np.linspace(-6, 6, 200),
    log_prior=None,
    n_classes=5,
    weight_mode="one",
    ess_value=None,
):
    theta_grid = np.asarray(theta_grid, dtype=np.float64)
    N_theta = theta_grid.size

    if log_prior is None:
        log_prior = np.zeros(N_theta, dtype=np.float64)

    total_log_like = np.zeros(N_theta, dtype=np.float64)

    for item_id, item_rows in group_df.groupby("item_id"):
        beta_center = item_difficulty_map.get(item_id, item_difficulty_map.get(str(item_id), 0.0))
        if item_id not in item_difficulty_map and str(item_id) not in item_difficulty_map:
            print(f"Warning: item_id {item_id} not found in item_difficulty_map; using 0.0")

        eta = theta_grid - beta_center  # (N_theta,)

        for _, row in item_rows.iterrows():
            rater_id = row["rater_id"]
            if rater_id not in judge_params:
                continue

            r_idx = judge_params[rater_id]["rater_idx"]
            cuts = cutpoints[r_idx]  # (K-1,)
            logp_all = compute_ordered_logprob_allcats(eta, cuts)  # (N_theta, K)

            q, w = row_to_soft_observation(
                row,
                n_classes=n_classes,
                weight_mode=weight_mode,
                ess_value=ess_value,
            )

            total_log_like += w * (logp_all @ q)

    unnormalized = total_log_like + log_prior
    unnormalized -= logsumexp(unnormalized)
    return np.exp(unnormalized)