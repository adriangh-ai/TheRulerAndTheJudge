import ast
import numpy as np
import warnings

from itertools import combinations
from .estimate import predict_human_item_distribution
from typing import Dict, Any


def jsd(p, q):
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    p = p / p.sum()
    q = q / q.sum()
    m = 0.5*(p+q)
    def kl(a,b):
        a = np.clip(a, 1e-15, 1.0)
        b = np.clip(b, 1e-15, 1.0)
        return np.sum(a*np.log(a/b))
    return 0.5*kl(p,m) + 0.5*kl(q,m)


def emd_ordinal_cost(p, q, d=None, eps=1e-12):
    """
    1D EMD / Wasserstein-1 for ordered discrete categories with *adjacent* costs.

    """
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)

    p = p / (p.sum() + eps)
    q = q / (q.sum() + eps)

    K = p.shape[0]
    if d is None:
        d = np.ones(K - 1, dtype=np.float64)
    else:
        d = np.asarray(d, dtype=np.float64)
        assert d.shape[0] == K - 1, "d must have length K-1"
        assert np.all(d >= 0), "d must be nonnegative"

    cdf_diff = np.abs(np.cumsum(p) - np.cumsum(q))[:-1]  # boundaries 1..K-1
    return float(np.dot(cdf_diff, d))


def category_positions_from_cutpoints(tau):
    """
    Map (K-1) ordered cutpoints to K category locations on the same latent axis.
    Uses midpoints for interior categories, and extrapolates ends.
    """
    tau = np.asarray(tau, dtype=np.float64)
    K = tau.shape[0] + 1
    x = np.empty(K, dtype=np.float64)

    if K == 2:
        # one cutpoint only: pick symmetric positions
        x[0] = tau[0] - 0.5
        x[1] = tau[0] + 0.5
        return x

    # interior category locations = midpoints between adjacent cutpoints
    x[1:-1] = 0.5 * (tau[:-1] + tau[1:])

    # extrapolate endpoints using nearest gap
    first_gap = tau[1] - tau[0]
    last_gap  = tau[-1] - tau[-2]
    x[0]  = tau[0]  - 0.5 * first_gap
    x[-1] = tau[-1] + 0.5 * last_gap
    return x


def adjacent_costs_from_cutpoints(tau, normalize="mean1"):
    """
    Returns d_k = x_{k+1} - x_k from cutpoints, with optional normalization.
    normalize:
      - None: raw costs in logit units
      - "mean1": mean(d)=1
      - "sumK-1": sum(d)=K-1
    """
    x = category_positions_from_cutpoints(tau)
    d = np.diff(x)
    # safety
    if np.any(d <= 0):
        raise ValueError("Non-increasing category positions; check cutpoints ordering/constraints.")

    if normalize == "mean1":
        d = d / d.mean()
    elif normalize == "sumK-1":
        d = d * ((len(d)) / d.sum())
    elif normalize is None:
        pass
    else:
        raise ValueError("Unknown normalize option.")
    return d


def _freq_vec(scores, K=5):
    v = np.zeros(K, dtype=np.float64)
    for s in scores:
        s = int(s)
        if 1 <= s <= K:
            v[s-1] += 1
    if v.sum() == 0:
        return v
    return v / v.sum()


def _parse_score_list(x):
    if x is None:
        return None
    if isinstance(x, float) and np.isnan(x):
        return None
    if isinstance(x, str):
        x = x.strip()
        if not x:
            return None
        try:
            x = ast.literal_eval(x)
        except Exception:
            return None
    if isinstance(x, (list, tuple, np.ndarray)):
        return list(x)
    return None


def _row_distribution(row, K=5):
    """
    Returns a probability vector for one row.

    Priority:
    - if score_list exists -> use full empirical distribution
    - else -> use one-hot(score)
    """
    score_list = _parse_score_list(row.get("score_list", None))

    if score_list is not None and len(score_list) > 0:
        counts = np.zeros(K, dtype=np.float64)
        for s in score_list:
            if np.isnan(s): continue
            s = int(s)
            if 1 <= s <= K:
                counts[s - 1] += 1.0
        if counts.sum() > 0:
            return counts / counts.sum()

    # fallback to hard label
    v = np.zeros(K, dtype=np.float64)
    s = int(row["score"])
    if 1 <= s <= K:
        v[s - 1] = 1.0
    return v


def _item_distribution(group_df, K=5, weighting="equal"):
    """
    Aggregate row-level distributions into one item-level distribution.

    weighting:
    - "equal": each row contributes equally
    - "count": rows with longer score_list get more weight

    Usually "equal" is the safer choice if score_list is repeated sampling
    from the same judge/item, because it avoids overweighting rows with larger T.
    """
    row_dists = []
    weights = []

    for _, row in group_df.iterrows():
        q = _row_distribution(row, K=K)
        if q.sum() == 0:
            continue

        row_dists.append(q)

        if weighting == "count":
            score_list = _parse_score_list(row.get("score_list", None))
            w = len(score_list) if score_list is not None and len(score_list) > 0 else 1.0
        else:
            w = 1.0

        weights.append(float(w))

    if not row_dists:
        return np.zeros(K, dtype=np.float64)

    row_dists = np.vstack(row_dists)      # (n_rows, K)
    weights = np.asarray(weights)         # (n_rows,)
    q = (row_dists * weights[:, None]).sum(axis=0)

    if q.sum() == 0:
        return q
    return q / q.sum()


def evaluate_system_item_jsd_emd(
    llm_sys_df,   # item_id, rater_id, score, optional score_list
    hum_sys_df,   # item_id, rater_id, score, optional score_list
    theta_grid,
    post_theta,
    item_map,
    human_cuts,
    human_rho_map=None,
    K=5,
    weighting="equal",   # "equal" or "count"
) -> Dict[str, Any]:
    """
    Mean JSD/EMD across items for:
      - pred: posterior-predictive human distribution
      - raw:  raw LLM item distribution
      - gold: human item distribution
    """

    human_rho_values = list(human_rho_map.values()) if human_rho_map else None
    d = adjacent_costs_from_cutpoints(human_cuts, normalize=None)

    gold_by_item = {
        item: _item_distribution(g, K=K, weighting=weighting)
        for item, g in hum_sys_df.groupby("item_id")
    }

    raw_by_item = {
        item: _item_distribution(g, K=K, weighting=weighting)
        for item, g in llm_sys_df.groupby("item_id")
    }

    jsd_pred_list, emd_pred_list = [], []
    jsd_raw_list,  emd_raw_list  = [], []

    for item_id, gold in gold_by_item.items():
        if gold.sum() == 0:
            continue

        raw = raw_by_item.get(item_id, None)
        if raw is None or raw.sum() == 0:
            continue

        beta_i = float(item_map.get(item_id, item_map.get(str(item_id), 0.0)))

        pred = predict_human_item_distribution(
            theta_grid=theta_grid,
            theta_posterior=post_theta,
            beta_i=beta_i,
            human_cutpoints=human_cuts,
            human_rho_values=human_rho_values,
        )

        jsd_pred_list.append(jsd(gold, pred))
        emd_pred_list.append(emd_ordinal_cost(gold, pred, d))

        jsd_raw_list.append(jsd(gold, raw))
        emd_raw_list.append(emd_ordinal_cost(gold, raw, d))

    def _mean(x):
        return float(np.mean(x)) if len(x) else np.nan

    return {
        "jsd_pred": _mean(jsd_pred_list),
        "emd_pred": _mean(emd_pred_list),
        "jsd_raw":  _mean(jsd_raw_list),
        "emd_raw":  _mean(emd_raw_list),
        "n_items_eval": int(len(jsd_pred_list)),
    }

# Generative pairwise probabilities and margin-based undecided baseline
def sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return 1.0 / (1.0 + np.exp(-x))
# Kept as implemented in the reported experiments.

def compute_pairwise_generative_probabilities_vectorized(
    posteriors_by_system: dict,
    theta_grid: np.ndarray,
    return_matrix: bool = False,
) -> dict:
    """
    Compute posterior expected generative pairwise probabilities
    g_ab = E[sigmoid(theta_a - theta_b)]
    for all unordered system pairs, using a vectorized implementation.

    Parameters
    ----------
    posteriors_by_system : dict
        Mapping {system_id: posterior_mass_vector}, all on the same theta_grid.
    theta_grid : np.ndarray, shape (G,)
        Shared latent grid.
    return_matrix : bool, default=False
        If True, also return the full S x S matrix of pairwise generative probs.

    Returns
    -------
    dict
        Mapping {(a, b): g_ab} for ordered keys with a < b under sorted system ids.

    If return_matrix=True, also returns:
    -------
    pairwise_g : dict
    g_matrix : np.ndarray, shape (S, S)
        Full matrix where g_matrix[i, j] = E[sigmoid(theta_i - theta_j)].
    system_ids : list[str]
        System ordering corresponding to rows/cols of g_matrix.
    """
    system_ids = sorted(posteriors_by_system.keys())
    theta_grid = np.asarray(theta_grid, dtype=np.float64)

    # Stack posterior masses: P.shape = (S, G)
    P = np.stack(
        [np.asarray(posteriors_by_system[s], dtype=np.float64) for s in system_ids],
        axis=0,
    )

    if P.ndim != 2:
        raise ValueError("Each posterior must be a 1D vector on the shared theta grid.")

    if P.shape[1] != theta_grid.shape[0]:
        raise ValueError("Posterior vectors must have the same length as theta_grid.")

    # Normalize defensively
    row_sums = P.sum(axis=1, keepdims=True)
    if np.any(row_sums <= 0):
        raise ValueError("Each posterior mass vector must sum to a positive value.")
    P = P / row_sums

    # Kernel K[u, v] = sigmoid(theta_u - theta_v)
    delta = theta_grid[:, None] - theta_grid[None, :]
    K = sigmoid(delta)  # shape (G, G)

    # Full pairwise matrix: G = P K P^T
    g_matrix = P @ K @ P.T  # shape (S, S)

    # Numerical clipping
    g_matrix = np.clip(g_matrix, 0.0, 1.0)

    pairwise_g = {}
    for i, j in combinations(range(len(system_ids)), 2):
        pairwise_g[(system_ids[i], system_ids[j])] = float(g_matrix[i, j])

    if return_matrix:
        return pairwise_g, g_matrix, system_ids
    return pairwise_g


def compute_nsmf_from_generative_probabilities(
    human_pairwise_g: dict,
    judge_pairwise_g: dict,
    check_decomposition: bool = True,
    atol: float = 1e-12,
    zero_tol: float = 1e-15,
) -> dict:
    """
    Compute Normalized Signed-Margin Fidelity (NSMF) from pairwise generative
    probabilities and its exact decomposition.

    Let:
      d_h = 2 g_h - 1
      d_j = 2 g_j - 1

    Then:
      sign(d) = preferred direction
      |d|     = predictive preference strength

    Per-pair error:
      (d_j - d_h)^2

    Exact decomposition:
      (d_j - d_h)^2
        = (c_j - c_h)^2 + 2 c_j c_h (1 - s_j s_h)

    where c = |d| and s = sign(d).

    NSMF uses the zero-preference baseline:
      baseline = mean(d_h^2)

      NSMF = 1 - mean((d_j - d_h)^2) / mean(d_h^2)

    Returns
    -------
    dict
        {
            "nsmf": float,
            "signed_margin_mse": float,
            "signed_margin_baseline": float,
            "uncertainty_mse": float,
            "direction_mse": float,
            "uncertainty_loss": float,
            "direction_loss": float,
            "n_pairs": int,
        }
    """
    if human_pairwise_g.keys() != judge_pairwise_g.keys():
        raise ValueError("human_pairwise_g and judge_pairwise_g must have identical pair keys.")

    if len(human_pairwise_g) == 0:
        return {
            "nsmf": np.nan,
            "signed_margin_mse": np.nan,
            "signed_margin_baseline": np.nan,
            "uncertainty_mse": np.nan,
            "direction_mse": np.nan,
            "uncertainty_loss": np.nan,
            "direction_loss": np.nan,
            "n_pairs": 0,
        }

    margin_mse_sum = 0.0
    uncertainty_mse_sum = 0.0
    direction_mse_sum = 0.0
    human_margins = []
    n = 0

    for pair, g_h in human_pairwise_g.items():
        g_j = judge_pairwise_g[pair]

        if not (0.0 <= g_h <= 1.0):
            warnings.warn(f"Human generative probability for pair {pair} is outside [0,1]: {g_h}")
            g_h = float(np.clip(g_h, 0.0, 1.0))
        if not (0.0 <= g_j <= 1.0):
            warnings.warn(f"Judge generative probability for pair {pair} is outside [0,1]: {g_j}")
            g_j = float(np.clip(g_j, 0.0, 1.0))

        d_h = 2.0 * g_h - 1.0
        d_j = 2.0 * g_j - 1.0

        if abs(d_h) <= zero_tol:
            d_h = 0.0
        if abs(d_j) <= zero_tol:
            d_j = 0.0

        c_h = abs(d_h)
        c_j = abs(d_j)
        # Note: np.sign(0) returns 0. This is mathematically safe here 
        # because the direction_term is multiplied by c_h and c_j, 
        # zeroing out the term correctly when either margin is exactly 0.
        s_h = float(np.sign(d_h))
        s_j = float(np.sign(d_j))

        margin_term = (d_j - d_h) ** 2
        uncertainty_term = (c_j - c_h) ** 2
        direction_term = 2.0 * c_j * c_h * (1.0 - s_j * s_h)

        margin_mse_sum += margin_term
        uncertainty_mse_sum += uncertainty_term
        direction_mse_sum += direction_term
        human_margins.append(d_h)
        n += 1

        if check_decomposition:
            if not np.isclose(margin_term, uncertainty_term + direction_term, atol=atol, rtol=0.0):
                raise AssertionError(
                    f"Decomposition failed for pair {pair}: "
                    f"margin_term={margin_term}, "
                    f"uncertainty_term+direction_term={uncertainty_term + direction_term}"
                )

    signed_margin_mse = margin_mse_sum / n
    uncertainty_mse = uncertainty_mse_sum / n
    direction_mse = direction_mse_sum / n

    if check_decomposition:
        if not np.isclose(signed_margin_mse, uncertainty_mse + direction_mse, atol=atol, rtol=0.0):
            raise AssertionError(
                f"Mean decomposition failed: "
                f"signed_margin_mse={signed_margin_mse}, "
                f"uncertainty_mse+direction_mse={uncertainty_mse + direction_mse}"
            )

    human_margins = np.asarray(human_margins, dtype=np.float64)
    signed_margin_baseline = float(np.mean(human_margins ** 2))

    if signed_margin_baseline <= zero_tol:
        nsmf = np.nan
        uncertainty_loss = np.nan
        direction_loss = np.nan
    else:
        uncertainty_loss = uncertainty_mse / signed_margin_baseline
        direction_loss = direction_mse / signed_margin_baseline
        nsmf = 1.0 - signed_margin_mse / signed_margin_baseline

        if check_decomposition:
            if not np.isclose(
                nsmf,
                1.0 - uncertainty_loss - direction_loss,
                atol=atol,
                rtol=0.0,
            ):
                raise AssertionError(
                    f"NSMF decomposition failed: "
                    f"nsmf={nsmf}, expected={1.0 - uncertainty_loss - direction_loss}"
                )

    return {
        "nsmf": float(nsmf) if not np.isnan(nsmf) else np.nan,
        "signed_margin_mse": float(signed_margin_mse),
        "signed_margin_baseline": float(signed_margin_baseline),
        "uncertainty_mse": float(uncertainty_mse),
        "direction_mse": float(direction_mse),
        "uncertainty_loss": float(uncertainty_loss) if not np.isnan(uncertainty_loss) else np.nan,
        "direction_loss": float(direction_loss) if not np.isnan(direction_loss) else np.nan,
    }


# Pareto dominance and skill scores for pairwise probabilities
def pairwise_dominance_probability(
    posterior_a: np.ndarray,
    posterior_b: np.ndarray,
    theta_grid: np.ndarray,
) -> float:
    """
    Compute the tie-adjusted dominance probability from two discrete posterior
    mass functions defined on the same theta grid.

    Parameters
    ----------
    posterior_a : np.ndarray, shape (G,)
        Posterior mass for system a over theta_grid.
    posterior_b : np.ndarray, shape (G,)
        Posterior mass for system b over theta_grid.
    theta_grid : np.ndarray, shape (G,)
        Shared latent grid. Must be sorted ascending.

    Returns
    -------
    float
        Mid-dominance probability P(theta_a > theta_b) + 0.5 P(theta_a = theta_b).
    """
    posterior_a = np.asarray(posterior_a, dtype=np.float64)
    posterior_b = np.asarray(posterior_b, dtype=np.float64)
    theta_grid = np.asarray(theta_grid, dtype=np.float64)

    if posterior_a.shape != posterior_b.shape or posterior_a.shape != theta_grid.shape:
        raise ValueError("posterior_a, posterior_b, and theta_grid must have the same shape.")

    # Normalize defensively
    sum_a = posterior_a.sum()
    sum_b = posterior_b.sum()

    if sum_a <= 0 or sum_b <= 0:
        raise ValueError("Posterior mass must sum to a positive value.")

    posterior_a = posterior_a / sum_a
    posterior_b = posterior_b / sum_b

    # Outer product of posterior masses.
    joint = np.outer(posterior_a, posterior_b)

    # Mid-dominance convention for a discretized latent grid:
    # P(a > b) + 0.5 P(a = b). This assigns identical posteriors a neutral
    # probability of 0.5 and preserves p_ab + p_ba = 1 exactly.
    gt_mask = theta_grid[:, None] > theta_grid[None, :]
    p_gt = float(joint[gt_mask].sum())
    p_eq = float(np.dot(posterior_a, posterior_b))
    return float(p_gt + 0.5 * p_eq)


def compute_pairwise_probabilities(
        posteriors_by_system: dict,
        theta_grid: np.ndarray,
    ) -> dict:
    """
    Compute pairwise dominance probabilities for all system pairs.

    Parameters
    ----------
    posteriors_by_system : dict
        Mapping {system_id: posterior_mass_vector}.
    theta_grid : np.ndarray
        Shared latent grid.

    Returns
    -------
    dict
        Mapping {(a, b): p_ab} for ordered keys with a < b under sorted system ids,
        where p_ab = P(theta_a > theta_b) + 0.5 P(theta_a = theta_b).
    """
    system_ids = sorted(posteriors_by_system.keys())
    pairwise_p = {}

    for a, b in combinations(system_ids, 2):
        p_ab = pairwise_dominance_probability(
            posteriors_by_system[a],
            posteriors_by_system[b],
            theta_grid
        )
        pairwise_p[(a, b)] = p_ab

    return pairwise_p


def compute_false_separation_rate(
    human_pairwise_p: dict,
    judge_pairwise_p: dict,
    tau_h: float = 0.8,
    tau_l: float = 0.8,
) -> float:
    """
    Compute False Separation Rate (FSR).

    A pair is human-indistinguishable if:
      |2 p_h - 1| < tau_h

    A judge falsely separates that pair if:
      |2 p_j - 1| >= tau_l

    Parameters
    ----------
    human_pairwise_p : dict
        Mapping {(a, b): p_h_ab}.
    judge_pairwise_p : dict
        Mapping {(a, b): p_j_ab}.
    tau_h : float
        Threshold for human distinguishability.
    tau_l : float
        Threshold for judge confident separation.

    Returns
    -------
    float
        False separation rate in [0, 1], or np.nan if there are no
        human-indistinguishable pairs.
    """
    if human_pairwise_p.keys() != judge_pairwise_p.keys():
        raise ValueError("human_pairwise_p and judge_pairwise_p must have identical pair keys.")

    if not (0.0 <= tau_h <= 1.0 and 0.0 <= tau_l <= 1.0):
        raise ValueError("tau_h and tau_l must be in [0, 1].")

    false_sep = 0
    n_human_indist = 0

    for pair, p_h in human_pairwise_p.items():
        p_j = judge_pairwise_p[pair]

        omega_h = abs(2.0 * p_h - 1.0)
        omega_j = abs(2.0 * p_j - 1.0)

        if omega_h < tau_h:
            n_human_indist += 1
            false_sep += int(omega_j >= tau_l)

    if n_human_indist == 0:
        return np.nan

    return float(false_sep / n_human_indist)