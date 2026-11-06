from .runtime import configure_runtime

# Preserve the original direct-import behavior while centralizing its definition.
configure_runtime()

import numpy as np
import pandas as pd
import pymc as pm
import pytensor
import pytensor.tensor as pt


def build_mfrm_model(
    df: pd.DataFrame, 
    meta, 
    fixed_cutpoints_list, 
    facet_sigma=0.5, 
    *args, **kwargs
):
    """Build the Stage-1 many-facet Rasch ordered-logistic model."""
    # 1. Extract Data
    system_idx = df["system_idx"].values
    item_idx   = df["item_idx"].values
    y_ord      = df["y_ord"].values
    rater_idx  = df["rater_idx"].values
    
    coords = {
        "system": meta["system_categories"],
        "items":  meta["item_categories"],
        "human_raters": meta["rater_categories"],
    }
    with pm.Model(coords=coords) as model:
        # System quality.
        sigma_theta = pm.HalfNormal("sigma_theta", sigma=facet_sigma)
        theta_z = pm.Normal("theta_z", mu=0, sigma=1.0, dims=("system"))
        theta   = pm.Deterministic("theta", theta_z * sigma_theta)
        
        # Item difficulty, zero-sum identified.
        sigma_beta = pm.HalfNormal("sigma_beta", sigma=facet_sigma)
        beta_z = pm.ZeroSumNormal("beta_z", sigma=1.0, dims=("items"))
        beta   = pm.Deterministic("beta", beta_z * sigma_beta)
        
        # Human-rater severity, zero-sum identified.
        sigma_rho_h = pm.HalfNormal("sigma_rho_h", sigma=facet_sigma)
        rho_h_raw = pm.ZeroSumNormal("rho_h_raw", sigma=1.0, dims=("human_raters"))
        rho_human = pm.Deterministic("rho_human", rho_h_raw * sigma_rho_h)
        
        eta = theta[system_idx] - beta[item_idx] - rho_human[rater_idx]
        
        cutpoints = pm.Normal(
            "cutpoints",
            mu=fixed_cutpoints_list, 
            sigma=3,  # Allow flexibility
            transform=pm.distributions.transforms.ordered, # Keep them ordered
            shape=fixed_cutpoints_list.shape
        )
        pm.OrderedLogistic(
            "y_ord",
            eta=eta,
            cutpoints=cutpoints,
            observed=y_ord
        )

    return model, coords


def _ordered_logistic_probs_from_eta(eta_q, cutpoints_obs):
    """
    eta_q: (Q, N)
    cutpoints_obs: (N, K-1)
    returns: probs (Q, N, K)
    """
    cdf = pm.math.sigmoid(cutpoints_obs[None, :, :] - eta_q[:, :, None])  # (Q, N, K-1)

    left = pt.concatenate(
        [pt.zeros((eta_q.shape[0], eta_q.shape[1], 1)), cdf],
        axis=2
    )
    right = pt.concatenate(
        [cdf, pt.ones((eta_q.shape[0], eta_q.shape[1], 1))],
        axis=2
    )
    probs = right - left
    probs = probs + 1e-9
    probs = probs / probs.sum(axis=2, keepdims=True)
    return probs


def build_llm_ord_model(
    df: pd.DataFrame,
    meta,
    eta_anchor_mean,
    eta_anchor_sd,
    human_cutpoints_mean,
    q_obs,                  # (N, K)
    obs_weight=None,        # (N,)
    facet_sigma=1.0,
    quadrature_order=7,
    add_alpha=False,
    weighted_soft_labels=True,
):
    """Build the Gauss--Hermite mean/SD approximation to Stage 2.

    This alternative is retained for sensitivity work. The paper-default path
    uses :func:`build_llm_ord_model_mc`.
    """
    system_idx = df["system_idx"].values
    item_idx   = df["item_idx"].values
    rater_idx  = df["rater_idx"].values

    n_classes = meta["n_classes"]
    N = len(df)

    q_obs = np.asarray(q_obs, dtype=np.float32)
    if q_obs.shape != (N, n_classes):
        raise ValueError(f"q_obs must have shape {(N, n_classes)}, got {q_obs.shape}")

    row_sums = q_obs.sum(axis=1)
    if not np.allclose(row_sums, 1.0, atol=1e-5):
        raise ValueError("Each row of q_obs must sum to 1.")

    if weighted_soft_labels or obs_weight is None:
        obs_weight = np.ones(N, dtype=np.float32)
    else:
        obs_weight = np.asarray(obs_weight, dtype=np.float32)
        if obs_weight.shape != (N,):
            raise ValueError(f"obs_weight must have shape {(N,)}, got {obs_weight.shape}")
        if np.any(obs_weight <= 0):
            raise ValueError("obs_weight must be strictly positive.")

    coords = {
        "system": meta["system_categories"],
        "items": meta["item_categories"],
        "llm_raters": meta["rater_categories"],
        "cutpoints": np.arange(n_classes - 1),
    }

    gh_x, gh_w = np.polynomial.hermite.hermgauss(quadrature_order)
    gh_x = np.asarray(np.sqrt(2.0) * gh_x, dtype=pytensor.config.floatX)
    gh_w = np.asarray(gh_w / np.sqrt(np.pi), dtype=pytensor.config.floatX)

    with pm.Model(coords=coords) as model:
        eta_mean = pm.Data(
            "eta_anchor_mean",
            np.asarray(eta_anchor_mean, dtype=pytensor.config.floatX),
            dims=("system", "items"),
        )
        eta_sd = pm.Data(
            "eta_anchor_sd",
            np.asarray(eta_anchor_sd, dtype=pytensor.config.floatX),
            dims=("system", "items"),
        )

        q_obs_data = pm.Data(
            "q_obs",
            np.asarray(q_obs, dtype=pytensor.config.floatX),
        )
        obs_weight_data = pm.Data(
            "obs_weight",
            np.asarray(obs_weight, dtype=pytensor.config.floatX),
        )

        eta_mean_obs = eta_mean[system_idx, item_idx]   # (N,)
        eta_sd_obs = eta_sd[system_idx, item_idx]       # (N,)

        init_cuts = np.tile(human_cutpoints_mean, (len(meta["rater_categories"]), 1))
        init_cuts = init_cuts.astype(pytensor.config.floatX)

        llm_cuts = pm.Normal(
            "llm_cuts",
            mu=np.asarray(human_cutpoints_mean, dtype=pytensor.config.floatX),
            sigma=2.0,
            dims=("llm_raters", "cutpoints"),
            transform=pm.distributions.transforms.ordered,
            initval=init_cuts,
        )

        cutpoints_obs = llm_cuts[rater_idx]  # (N, K-1)

        eta_q = eta_mean_obs[None, :] + eta_sd_obs[None, :] * gh_x[:, None]   # (Q, N)

        if add_alpha:
            log_alpha = pm.Normal("log_alpha", mu=0.0, sigma=0.35, dims=("llm_raters",))
            alpha_obs = pt.exp(log_alpha[rater_idx])    # (N,)
            eta_q = alpha_obs[None, :] * eta_q

        probs_q = _ordered_logistic_probs_from_eta(eta_q, cutpoints_obs)      # (Q, N, K)
        expected_probs = pt.sum(gh_w[:, None, None] * probs_q, axis=0)        # (N, K)

        expected_probs = expected_probs + 1e-12
        expected_probs = expected_probs / pt.sum(expected_probs, axis=1, keepdims=True)

        # Row-level distributional cross-entropy. With the paper default, each
        # judge--item row has unit weight and repeated score_list samples are
        # represented by q_obs rather than treated as independent replicates.
        row_logp = pt.sum(q_obs_data * pt.log(expected_probs), axis=1)   # (N,)
        pm.Potential("soft_label_loglik", pt.sum(obs_weight_data * row_logp))

    return model, coords


def build_llm_ord_model_mc(
    df: pd.DataFrame,
    meta,
    eta_anchor_draws,        # (M, S, I)
    human_cutpoints_mean,
    q_obs,                   # (N, K)
    obs_weight=None,         # (N,)
    add_alpha=False,
    weighted_soft_labels=True,
):
    """Build the paper's Stage-2 Monte Carlo anchored ordinal model.

    Stage-1 uncertainty is integrated observation-wise: category probabilities
    for each judge observation are averaged over the selected joint Stage-1
    posterior draws before the row-level distributional likelihood is evaluated.

    ``weighted_soft_labels=True`` is the historical option name for the paper
    convention: every row receives unit likelihood weight. A hard ``score``
    therefore defines a one-hot categorical target, while ``score_list`` defines
    one empirical target distribution for that row.
    """
    system_idx = df["system_idx"].values
    item_idx   = df["item_idx"].values
    rater_idx  = df["rater_idx"].values

    n_classes = meta["n_classes"]
    N = len(df)

    q_obs = np.asarray(q_obs, dtype=np.float32)
    if q_obs.shape != (N, n_classes):
        raise ValueError(f"q_obs must have shape {(N, n_classes)}, got {q_obs.shape}")

    if weighted_soft_labels or obs_weight is None:
        obs_weight = np.ones(N, dtype=np.float32)
    else:
        obs_weight = np.asarray(obs_weight, dtype=np.float32)
        if obs_weight.shape != (N,):
            raise ValueError(f"obs_weight must have shape {(N,)}, got {obs_weight.shape}")
        if np.any(obs_weight <= 0):
            raise ValueError("obs_weight must be strictly positive.")

    coords = {
        "system": meta["system_categories"],
        "items": meta["item_categories"],
        "llm_raters": meta["rater_categories"],
        "cutpoints": np.arange(n_classes - 1),
    }

    with pm.Model(coords=coords) as model:
        eta_draws = pm.Data(
            "eta_anchor_draws",
            np.asarray(eta_anchor_draws, dtype=pytensor.config.floatX),
        )  # (M, S, I)

        q_obs_data = pm.Data("q_obs", q_obs.astype(pytensor.config.floatX))
        obs_weight_data = pm.Data("obs_weight", obs_weight.astype(pytensor.config.floatX))

        eta_mn = eta_draws[:, system_idx, item_idx]   # (M, N)

        init_cuts = np.tile(human_cutpoints_mean, (len(meta["rater_categories"]), 1))
        init_cuts = init_cuts.astype(pytensor.config.floatX)

        llm_cuts = pm.Normal(
            "llm_cuts",
            mu=np.asarray(human_cutpoints_mean, dtype=pytensor.config.floatX),
            sigma=2.0,
            dims=("llm_raters", "cutpoints"),
            transform=pm.distributions.transforms.ordered,
            initval=init_cuts,
        )

        cutpoints_obs = llm_cuts[rater_idx]   # (N, K-1)

        if add_alpha:
            log_alpha = pm.Normal("log_alpha", mu=0.0, sigma=0.35, dims=("llm_raters",))
            alpha_obs = pt.exp(log_alpha[rater_idx])         # (N,)
            eta_mn = alpha_obs[None, :] * eta_mn

        probs_m = _ordered_logistic_probs_from_eta(eta_mn, cutpoints_obs)   # (M, N, K)
        log_probs_m = pt.log(probs_m + 1e-12)                                # (M, N, K)

        log_avg_probs = pm.math.logsumexp(log_probs_m, axis=0) - np.log(eta_anchor_draws.shape[0])  # (N, K)

        row_logp = pt.sum(q_obs_data * log_avg_probs, axis=1)
        pm.Potential("soft_label_loglik", pt.sum(obs_weight_data * row_logp))

    return model, coords


def build_llm_ord_dif_model_meanfield(
    df: pd.DataFrame, 
    meta,
    human_beta,    # From Stage 1 (Fixed)
    human_theta,   # From Stage 1 (Fixed)
    human_beta_std, # From Stage 1 (Informative Prior)
    human_theta_std, # From Stage 1 (Informative Prior)
    human_cutpoints_mean, # Optional: Use mean human cuts as a soft prior
    facet_sigma=1.0,
):
    """
    Legacy plug-in DIF variant (single active definition retained).

    Conditions on Stage-1 posterior-mean theta/beta and compensates for the
    discarded Stage-1 uncertainty with the logit-scale variance-matching
    correction (lambda^2 = 3/pi^2), applied symmetrically to eta and the
    cutpoints. Approximations relative to build_llm_ord_dif_model (MC variant):
      (i)   matches the marginal category probability approximately rather
            than marginalizing the likelihood exactly;
      (ii)  combines marginal theta/beta SDs under an independence assumption,
            discarding Stage-1 posterior correlations;
      (iii) the per-observation rescaling places beta_diff and llm_cuts on a
            shrunken, observation-specific scale, so their magnitudes are not
            directly comparable to the main Stage-2 model's logits.
    Retained for comparison against the MC-marginalized variant.
    """
    # 1. Extract Indices
    system_idx = df["system_idx"].values
    item_idx   = df["item_idx"].values
    rater_idx  = df["rater_idx"].values
    y_ord      = df["y_ord"].values
    
    n_classes = 5
    coords = {
        "system": meta["system_categories"],
        "items":  meta["item_categories"],
        "llm_raters": meta["rater_categories"],
        "cutpoints": np.arange(n_classes - 1)
    }

    with pm.Model(coords=coords) as model:
        theta = pm.Data("theta", human_theta, dims=("system"))
        beta  = pm.Data("beta", human_beta, dims=("items"))
        theta_std = pm.Data("theta_std", human_theta_std, dims=("system"))
        beta_std  = pm.Data("beta_std", human_beta_std, dims=("items"))

        beta_diff = pm.ZeroSumNormal(
            "beta_diff", sigma=facet_sigma,
            dims=("llm_raters", "items")
        )

        # Heteroskedastic correction (logit-scale variance-matching; lambda^2 = 3/pi^2)
        # Uses marginal Stage 1 SDs combined via independence assumption, since the
        # DIF model is passed human_theta_std and human_beta_std rather than paired draws.
        C_LOGISTIC = 3.0 / (np.pi ** 2)
        tau_sq = theta_std[system_idx] ** 2 + beta_std[item_idx] ** 2
        correction = pm.math.sqrt(1.0 + C_LOGISTIC * tau_sq)

        eta_raw = (
            theta[system_idx]
            - beta[item_idx]
            + beta_diff[rater_idx, item_idx]
        )

        # Symmetric rescaling of (cutpoints - eta) numerator
        eta = eta_raw / correction
        llm_cuts_raw = pm.Normal(
            "llm_cuts",
            mu=human_cutpoints_mean,   # Soft anchor
            sigma=2,                   # Allow deviation for per-judge calibration
            dims=("llm_raters", "cutpoints"),
            transform=pm.distributions.transforms.ordered,  # Enforce k1 < k2 < k3
            initval=np.tile(
                human_cutpoints_mean,
                (len(meta["rater_categories"]), 1),
            ).astype(pytensor.config.floatX),
        )

        # Per-observation cutpoints, then rescaled by the same correction
        cutpoints_obs = llm_cuts_raw[rater_idx] / correction[:, None]

        pm.OrderedLogistic(
            "y_ord",
            eta=eta,
            cutpoints=cutpoints_obs,
            observed=y_ord,
        )

    return model, coords


def build_llm_ord_dif_model(
    df: pd.DataFrame,
    meta,
    eta_anchor_draws,        # (M, S, I) joint Stage-1 posterior draws of theta_s - beta_i
    human_cutpoints_mean,
    q_obs,                   # (N, K)
    obs_weight=None,         # (N,)
    facet_sigma=1.0,
    add_alpha=False,
    weighted_soft_labels=True,
):
    """
    MC-marginalized judge-by-item DIF variant.

    This is exactly build_llm_ord_model_mc plus one ZeroSumNormal
    judge-by-item interaction term added to the latent predictor:

        eta_{m,n} = [theta_{s(n)} - beta_{i(n)}]_m + beta_diff[j(n), i(n)]

    with observation-wise category probabilities averaged over the SAME
    Stage-1 anchor draw subsample and the same soft-label Potential as the main
    Stage-2 calibration model. Within each draw, theta/beta dependence is
    retained. Consequences:
      - no variance-matching approximation is needed for the selected draw
        approximation;
      - beta_diff and llm_cuts live on the SAME latent logit scale as the
        main model, so DIF magnitudes are directly comparable to Stage-2
        cutpoint geometry and human-ruler quantities;
      - the DIF audit is a strict one-term extension of the reported
        calibration model — the cleanest framing for reporting it as a
        robustness/sensitivity result.

    ZeroSumNormal over the items axis (within each judge) prevents beta_diff
    from absorbing a global shift already captured by the judge cutpoints.
    """
    system_idx = df["system_idx"].values
    item_idx   = df["item_idx"].values
    rater_idx  = df["rater_idx"].values

    n_classes = meta["n_classes"]
    N = len(df)

    q_obs = np.asarray(q_obs, dtype=np.float32)
    if q_obs.shape != (N, n_classes):
        raise ValueError(f"q_obs must have shape {(N, n_classes)}, got {q_obs.shape}")

    if weighted_soft_labels or obs_weight is None:
        obs_weight = np.ones(N, dtype=np.float32)
    else:
        obs_weight = np.asarray(obs_weight, dtype=np.float32)
        if obs_weight.shape != (N,):
            raise ValueError(f"obs_weight must have shape {(N,)}, got {obs_weight.shape}")
        if np.any(obs_weight <= 0):
            raise ValueError("obs_weight must be strictly positive.")

    coords = {
        "system": meta["system_categories"],
        "items": meta["item_categories"],
        "llm_raters": meta["rater_categories"],
        "cutpoints": np.arange(n_classes - 1),
    }

    with pm.Model(coords=coords) as model:
        eta_draws = pm.Data(
            "eta_anchor_draws",
            np.asarray(eta_anchor_draws, dtype=pytensor.config.floatX),
        )  # (M, S, I)

        q_obs_data = pm.Data("q_obs", q_obs.astype(pytensor.config.floatX))
        obs_weight_data = pm.Data("obs_weight", obs_weight.astype(pytensor.config.floatX))

        eta_mn = eta_draws[:, system_idx, item_idx]   # (M, N)

        # Judge-by-item DIF, zero-sum over items within each judge.
        beta_diff = pm.ZeroSumNormal(
            "beta_diff",
            sigma=facet_sigma,
            dims=("llm_raters", "items"),
        )
        eta_mn = eta_mn + beta_diff[rater_idx, item_idx][None, :]   # (M, N)

        init_cuts = np.tile(human_cutpoints_mean, (len(meta["rater_categories"]), 1))
        init_cuts = init_cuts.astype(pytensor.config.floatX)

        llm_cuts = pm.Normal(
            "llm_cuts",
            mu=np.asarray(human_cutpoints_mean, dtype=pytensor.config.floatX),
            sigma=2.0,
            dims=("llm_raters", "cutpoints"),
            transform=pm.distributions.transforms.ordered,
            initval=init_cuts,
        )

        cutpoints_obs = llm_cuts[rater_idx]   # (N, K-1)

        if add_alpha:
            # Discrimination applies to the anchored latent signal plus DIF,
            # mirroring the main model's convention of scaling eta.
            log_alpha = pm.Normal("log_alpha", mu=0.0, sigma=0.35, dims=("llm_raters",))
            alpha_obs = pt.exp(log_alpha[rater_idx])         # (N,)
            eta_mn = alpha_obs[None, :] * eta_mn

        probs_m = _ordered_logistic_probs_from_eta(eta_mn, cutpoints_obs)   # (M, N, K)
        log_probs_m = pt.log(probs_m + 1e-12)                                # (M, N, K)

        log_avg_probs = pm.math.logsumexp(log_probs_m, axis=0) - np.log(eta_anchor_draws.shape[0])  # (N, K)

        row_logp = pt.sum(q_obs_data * log_avg_probs, axis=1)
        pm.Potential("soft_label_loglik", pt.sum(obs_weight_data * row_logp))

    return model, coords