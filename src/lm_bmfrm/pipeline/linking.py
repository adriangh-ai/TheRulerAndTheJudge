"""Latent-scale linking utilities."""
from __future__ import annotations
import numpy as np
from sklearn.linear_model import HuberRegressor


def fit_robust_affine_link(theta_fold_overlap: np.ndarray, theta_anchor_overlap: np.ndarray):
    x = np.asarray(theta_fold_overlap, dtype=float).reshape(-1, 1)
    y = np.asarray(theta_anchor_overlap, dtype=float).ravel()

    if len(y) < 2:
        raise ValueError("Need at least 2 overlap systems to fit affine link.")

    model = HuberRegressor().fit(x, y)
    b = float(model.coef_[0])
    a = float(model.intercept_)

    y_hat = a + b * x.ravel()
    rmse = float(np.sqrt(np.mean((y - y_hat) ** 2)))
    return a, b, rmse


def transform_posterior_to_anchor_grid(post_fold, theta_grid_fold, a, b, theta_grid_anchor):
    post_fold = np.asarray(post_fold, dtype=float)
    theta_grid_fold = np.asarray(theta_grid_fold, dtype=float)
    theta_grid_anchor = np.asarray(theta_grid_anchor, dtype=float)

    if b <= 0:
        raise ValueError(f"Link slope must be positive for ordered latent scale. Got b={b}")
    if post_fold.shape != theta_grid_fold.shape:
        raise ValueError("post_fold and theta_grid_fold must have the same shape.")

    s = post_fold.sum()
    if s <= 0:
        raise ValueError("post_fold has zero total mass.")
    post_fold = post_fold / s

    linked_grid = a + b * theta_grid_fold
    post_anchor = np.zeros_like(theta_grid_anchor, dtype=float)

    x_min = theta_grid_anchor[0]
    x_max = theta_grid_anchor[-1]

    for x, p in zip(linked_grid, post_fold):
        if p == 0.0:
            continue
        if x <= x_min:
            post_anchor[0] += p
            continue
        if x >= x_max:
            post_anchor[-1] += p
            continue

        idx = np.searchsorted(theta_grid_anchor, x)
        x0 = theta_grid_anchor[idx - 1]
        x1 = theta_grid_anchor[idx]
        w1 = (x - x0) / (x1 - x0)
        w0 = 1.0 - w1
        post_anchor[idx - 1] += p * w0
        post_anchor[idx] += p * w1

    s2 = post_anchor.sum()
    if s2 <= 0:
        raise ValueError("Linked posterior collapsed to zero mass on anchor grid.")
    return post_anchor / s2


def extract_anchor_system_means_by_dim(idata_anchor_by_dim, theta_var_name="theta"):
    out = {}
    for DIM, idata_dim in idata_anchor_by_dim.items():
        system_categories = [str(x) for x in idata_dim.posterior.coords["system"].values]
        theta_draws = np.asarray(idata_dim.posterior[theta_var_name]).reshape(-1, len(system_categories))
        means = theta_draws.mean(axis=0)
        out[DIM] = {str(system_categories[i]): float(means[i]) for i in range(len(system_categories))}
    return out
