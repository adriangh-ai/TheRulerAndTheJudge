"""Stage-1 posterior extraction and deterministic draw selection."""
from __future__ import annotations
import numpy as np
from scipy.spatial.distance import cdist


def extract_eta_anchor_draws(
    idata_anchor,
    target_system_categories,
    target_item_categories,
    theta_var_name="theta",
    beta_var_name="beta",
    anchor_draw_compute_grid=None,
):
    post = idata_anchor.posterior

    anchor_systems = [str(x) for x in post.coords["system"].values]
    anchor_items = [str(x) for x in post.coords["items"].values]

    system_lookup = {sid: i for i, sid in enumerate(anchor_systems)}
    item_lookup = {iid: i for i, iid in enumerate(anchor_items)}

    theta_draws = np.asarray(post[theta_var_name]).reshape(-1, len(anchor_systems))
    beta_draws = np.asarray(post[beta_var_name]).reshape(-1, len(anchor_items))

    if anchor_draw_compute_grid is not None:
        theta_draws = theta_draws[anchor_draw_compute_grid]
        beta_draws = beta_draws[anchor_draw_compute_grid]

    sys_idx = [system_lookup[str(s)] for s in target_system_categories]
    item_idx = [item_lookup[str(i)] for i in target_item_categories]

    theta_aligned = theta_draws[:, sys_idx]
    beta_aligned = beta_draws[:, item_idx]

    eta_draws = theta_aligned[:, :, None] - beta_aligned[:, None, :]
    return eta_draws.astype(np.float32)


def extract_eta_anchor_moments(
    idata_anchor,
    target_system_categories,
    target_item_categories,
    theta_var_name="theta",
    beta_var_name="beta",
):
    post = idata_anchor.posterior

    anchor_systems = [str(x) for x in post.coords["system"].values]
    anchor_items = [str(x) for x in post.coords["items"].values]

    system_lookup = {sid: i for i, sid in enumerate(anchor_systems)}
    item_lookup = {iid: i for i, iid in enumerate(anchor_items)}

    theta_draws = np.asarray(post[theta_var_name]).reshape(-1, len(anchor_systems))
    beta_draws = np.asarray(post[beta_var_name]).reshape(-1, len(anchor_items))

    sys_idx = [system_lookup[str(s)] for s in target_system_categories]
    item_idx = [item_lookup[str(i)] for i in target_item_categories]

    theta_aligned = theta_draws[:, sys_idx]
    beta_aligned = beta_draws[:, item_idx]

    eta_draws = theta_aligned[:, :, None] - beta_aligned[:, None, :]

    eta_mean = eta_draws.mean(axis=0).astype(np.float32)
    eta_sd = eta_draws.std(axis=0).astype(np.float32)

    return eta_mean, eta_sd


def _posterior_draw_feature_matrix(
    idata_anchor,
    theta_var_name="theta",
    beta_var_name="beta",
    cutpoints_var_name="cutpoints",
):
    """
    Build a standardized feature matrix over posterior draws for medoid selection.
    Features include theta, beta, and cutpoints posterior draws.
    """
    post = idata_anchor.posterior

    theta_draws = np.asarray(post[theta_var_name]).reshape(post.sizes["chain"] * post.sizes["draw"], -1)
    beta_draws = np.asarray(post[beta_var_name]).reshape(post.sizes["chain"] * post.sizes["draw"], -1)

    blocks = [theta_draws, beta_draws]
    if cutpoints_var_name in post:
        cut_draws = np.asarray(post[cutpoints_var_name]).reshape(post.sizes["chain"] * post.sizes["draw"], -1)
        blocks.append(cut_draws)

    X = np.concatenate(blocks, axis=1).astype(np.float64)
    mu = X.mean(axis=0, keepdims=True)
    sd = X.std(axis=0, keepdims=True)
    sd[sd < 1e-8] = 1.0
    X = (X - mu) / sd
    return X


def select_anchor_draw_indices_medoids(
    idata_anchor,
    n_select=64,
    random_state=42,
    max_iter=20,
):
    """
    Approximate k-medoids over posterior draws using alternating updates.
    Returns representative draw indices from the flattened (chain, draw) posterior.
    """
    X = _posterior_draw_feature_matrix(idata_anchor)
    n = X.shape[0]
    if n_select >= n:
        return np.arange(n, dtype=int)
    if n_select <= 0:
        raise ValueError("n_select must be positive.")

    rng = np.random.default_rng(random_state)
    D = cdist(X, X, metric="euclidean").astype(np.float64)

    # Farthest-point initialization
    medoids = [int(rng.integers(n))]
    min_dist = D[medoids[0]].copy()
    for _ in range(1, n_select):
        nxt = int(np.argmax(min_dist))
        medoids.append(nxt)
        min_dist = np.minimum(min_dist, D[nxt])
    medoids = np.array(medoids, dtype=int)

    for _ in range(max_iter):
        assign = np.argmin(D[:, medoids], axis=1)
        new_medoids = medoids.copy()
        used = set()
        for j in range(n_select):
            members = np.where(assign == j)[0]
            if len(members) == 0:
                candidates = np.setdiff1d(np.arange(n), np.array(list(used), dtype=int), assume_unique=False)
                if len(candidates) == 0:
                    continue
                chosen = int(candidates[np.argmax(np.min(D[candidates][:, medoids], axis=1))])
            else:
                intra = D[np.ix_(members, members)]
                chosen = int(members[np.argmin(intra.sum(axis=1))])
            new_medoids[j] = chosen
            used.add(chosen)

        new_medoids = np.unique(new_medoids)
        if len(new_medoids) < n_select:
            candidates = np.setdiff1d(np.arange(n), new_medoids, assume_unique=False)
            need = n_select - len(new_medoids)
            if need > 0:
                extra = candidates[np.argsort(-np.min(D[candidates][:, new_medoids], axis=1))[:need]]
                new_medoids = np.concatenate([new_medoids, extra])
        new_medoids = np.array(sorted(new_medoids[:n_select]), dtype=int)

        if np.array_equal(np.sort(medoids), np.sort(new_medoids)):
            medoids = new_medoids
            break
        medoids = new_medoids

    return np.array(sorted(medoids), dtype=int)


def select_anchor_draw_indices_linspace(idata_anchor, n_select=64):
    n = int(idata_anchor.posterior.sizes["chain"] * idata_anchor.posterior.sizes["draw"])
    if n_select >= n:
        return np.arange(n, dtype=int)
    return np.unique(np.linspace(0, n - 1, n_select, dtype=int))


def select_anchor_draw_indices(
    idata_anchor,
    n_select,
    method="medoids",
    random_state=42,
):
    method = method.lower()
    if method == "medoids":
        return select_anchor_draw_indices_medoids(
            idata_anchor=idata_anchor,
            n_select=n_select,
            random_state=random_state,
        )
    if method in {"linspace", "thin", "uniform"}:
        return select_anchor_draw_indices_linspace(
            idata_anchor=idata_anchor,
            n_select=n_select,
        )
    raise ValueError(f"Unknown anchor draw selection method: {method}")
