import ast
import warnings
from typing import Sequence

import numpy as np
import pandas as pd


from .cutpoints import alpha_per_item, prepare_cem_cutpoints


def _complete_missing_cutpoints(counts, cp_obs, n_classes, alpha=0.5, target_logit_range=6.0):
    """
    Complete missing ordinal cutpoints when some categories are absent.

    Strategy:
    - Build a full monotone surrogate set of K-1 thresholds from smoothed category frequencies.
    - Align that surrogate set to the observed CEM cutpoints on the identifiable thresholds.
    - Preserve the observed CEM cutpoints exactly where available.

    This is only for initialization, not for substantive interpretation.
    """
    counts = np.asarray(counts, dtype=float)
    cp_obs = np.asarray(cp_obs, dtype=float)

    K = n_classes
    if len(cp_obs) == K - 1:
        return cp_obs

    # Threshold j is directly identifiable only if both adjacent categories j and j+1 appear
    identified = np.flatnonzero((counts[:-1] > 0) & (counts[1:] > 0))

    # Smoothed cumulative-probability thresholds (finite even with empty categories)
    smoothed = counts + alpha
    cum_probs = np.cumsum(smoothed)[:-1] / smoothed.sum()
    cum_probs = np.clip(cum_probs, 1e-6, 1 - 1e-6)
    base = np.log(cum_probs / (1.0 - cum_probs))  # logit

    # Best case: number of observed cutpoints matches identifiable thresholds
    if len(cp_obs) == len(identified) and len(cp_obs) >= 2:
        X = np.vstack([np.ones_like(base[identified]), base[identified]]).T
        a, b = np.linalg.lstsq(X, cp_obs, rcond=None)[0]

        # If something pathological happens, fall back to centered/rescaled base
        if b <= 0:
            full = base - base.mean()
            span = full[-1] - full[0]
            if span > 0:
                full = full * (target_logit_range / span)
            return full

        full = a + b * base
        full[identified] = cp_obs  # preserve observed CEM thresholds exactly
        return full

    # Only one identifiable threshold: shift base to match it
    if len(cp_obs) == len(identified) == 1:
        full = base - base[identified[0]] + cp_obs[0]
        full[identified[0]] = cp_obs[0]
        return full

    # Fallback: use smoothed frequency thresholds, centered/rescaled
    warnings.warn(
        f"Could not reliably align observed cutpoints (got {len(cp_obs)}) "
        f"to identifiable thresholds (got {len(identified)}). "
        "Using smoothed frequency-based full initialization.",
        RuntimeWarning,
    )
    full = base - base.mean()
    span = full[-1] - full[0]
    if span > 0:
        full = full * (target_logit_range / span)
    return full


def prepare_mfrm_data(df: pd.DataFrame, get_cutpoints=True, n_classes=5):
    df = df.copy()

    # 1) Factorize facets -> integer indices
    df["item_idx"], item_categories = pd.factorize(df["item_id"])
    df["system_idx"], system_categories = pd.factorize(df["system_id"])
    df["rater_idx"], rater_categories = pd.factorize(df["rater_id"])
    df["dim_idx"], dim_categories = pd.factorize(df["dimension"])

    if df["score"].isna().any():
        raise ValueError("NaN scores present; drop/clean before prepare_mfrm_data.")

    s = df["score"].astype(int).values
    if s.min() < 1 or s.max() > n_classes:
        raise ValueError(f"Scores out of range [1,{n_classes}] found: min={s.min()}, max={s.max()}")

    df["y_ord"] = s - 1  # 0..4 always

    # score_maps for completeness / decoding
    score_maps = {dim: {k: k-1 for k in range(1, n_classes+1)} for dim in dim_categories}

    # 3) Cutpoints (still per dimension, but on the fixed K=5 scale)
    cutpoints = []
    if get_cutpoints:
        alpha_inter = df[~df["is_llm"]].copy().pivot_table(
            index=["item_id", "system_id", "dimension"], 
            columns="rater_id", 
            values="score"
        )
        alpha_inter["score_list"] = alpha_inter.apply(lambda row: row.dropna().tolist(), axis=1)
        alpha_inter["score_list"] = alpha_inter["score_list"].apply(lambda x: list(map(int, x)))
        alpha_inter.reset_index(inplace=True)

        for dim_name in dim_categories:
            dim_df = df[(df["dimension"] == dim_name) & (~df["is_llm"])].copy()

            _item_list = alpha_inter[alpha_inter["dimension"] == dim_name]["score_list"].tolist()
            _item_list = list(filter(lambda x: len(x) == 3, _item_list))
            _item_list = np.asarray(_item_list, dtype=np.float32).T  # shape (N_items,)
            _, cem = alpha_per_item(
                reliability_data=_item_list,
                level_of_measurement="ordinal",
                value_domain=np.arange(1, n_classes + 1)
            ) # should be (K,K) or similar

            cp = prepare_cem_cutpoints(cem, target_logit_range=6.0)  # (K-1,)
            # Safety: ensure length is K-1
            if len(cp) != (n_classes - 1):
                counts = (
                    dim_df["y_ord"]
                    .value_counts()
                    .reindex(range(n_classes), fill_value=0)
                    .sort_index()
                    .values
                )

                missing_classes = np.where(counts == 0)[0].tolist()

                warnings.warn(
                    f"Dimension '{dim_name}' has missing ordinal classes {missing_classes}. "
                    f"CEM returned {len(cp)} cutpoints instead of {n_classes - 1}. "
                    "Completing missing cutpoints with smoothed frequency-based initialization.",
                    RuntimeWarning,
                )

                cp = _complete_missing_cutpoints(
                    counts=counts,
                    cp_obs=cp,
                    n_classes=n_classes,
                    alpha=0.5,
                    target_logit_range=6.0,
                )
            cutpoints.append(cp)

    meta = {
        "item_categories": item_categories,
        "system_categories": system_categories,
        "rater_categories": rater_categories,
        "dim_categories": dim_categories,
        "score_maps": score_maps,
        "n_classes": n_classes,
    }

    return df, meta, cutpoints


def conformal_q(residuals, alpha=0.05):
    """
    Split conformal order-stat quantile: k = ceil((n+1)(1-alpha)).
    For n=17 and alpha=0.05, this becomes the max residual (conservative).
    """
    r = np.asarray(residuals, dtype=np.float64)
    r = r[~np.isnan(r)]
    n = len(r)
    if n == 0:
        return np.nan
    k = int(np.ceil((n + 1) * (1 - alpha)))
    k = min(k, n)
    return np.sort(r)[k - 1]


def _parse_score_list(x):
    """
    Normalize score_list input into a Python list, or None if unavailable.

    Accepts:
      - list / tuple / np.ndarray
      - stringified list like "[1, 2, 2, 3]"
      - None / NaN -> None
    """
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


def _counts_from_row(row, n_classes: int) -> np.ndarray:
    """
    Return raw class counts for one observation row.

    Priority:
      1) score_list, if present
      2) hard score column
    """
    counts = np.zeros(n_classes, dtype=np.float64)

    score_list = _parse_score_list(row.get("score_list", None))
    if score_list is not None and len(score_list) > 0:
        for s in score_list:
            if pd.isna(s):
                # NaN entries in repeated samples are ignored.
                continue
            s = int(s)
            if s < 1 or s > n_classes:
                warnings.warn(f"score_list contains out-of-range value {s} for n_classes={n_classes}")
                continue
            counts[s - 1] += 1.0
        return counts

    # fallback: hard label
    s = row.get("score", None)
    if pd.isna(s):
        raise ValueError("Row has neither valid score_list nor score.")
    s = int(s)
    if s < 1 or s > n_classes:
        raise ValueError(f"score out of range [1,{n_classes}] found: {s}")
    counts[s - 1] = 1.0
    return counts


def prepare_llm_judge_data(
    df: pd.DataFrame,
    n_classes: int = 5,
    aggregate_repeats: bool = False,
    group_cols: Sequence[str] = ("item_id", "system_id", "rater_id", "dimension"),
):
    """
    Prepare Stage-2 data for the soft-label ordinal judge model.

    Semantics
    ---------
    - Each row may represent either:
        (a) one hard judgment via `score`, or
        (b) multiple repeated judgments via `score_list`.
    - q_obs stores the normalized class distribution per observation.
    - obs_weight records the number of underlying judgments contributing to the row.
      The paper-default Stage-2 model deliberately gives every row unit likelihood
      weight: one hard score is categorical, whereas score_list is represented as
      one empirical target distribution. obs_weight is retained for explicit
      count-weighted sensitivity variants.

    If aggregate_repeats=True:
      repeated rows with the same `group_cols` are merged by summing raw counts.

    Returns
    -------
    df_obs : pd.DataFrame
        Observation table used by Stage 2.
    meta : dict
        Facet category metadata.
    q_obs : np.ndarray, shape (N, K)
        Soft target distributions.
    obs_weight : np.ndarray, shape (N,)
        Number of source judgments represented by each row.
    """
    df = df.copy()

    # Build raw counts row-by-row
    row_counts = np.vstack([_counts_from_row(row, n_classes) for _, row in df.iterrows()])
    counts_df = pd.DataFrame(
        row_counts,
        columns=[f"class_{k}" for k in range(1, n_classes + 1)],
        index=df.index,
    )

    df_counts = pd.concat([df.reset_index(drop=True), counts_df.reset_index(drop=True)], axis=1)

    class_cols = [f"class_{k}" for k in range(1, n_classes + 1)]

    if aggregate_repeats:
        # Sum counts across repeated outputs from same judge/system/item/dimension
        agg = (
            df_counts.groupby(list(group_cols), dropna=False)[class_cols]
            .sum()
            .reset_index()
        )
        df_obs = agg[list(group_cols)].copy()
        counts = agg[class_cols].to_numpy(np.float64)
    else:
        df_obs = df_counts[list(group_cols)].copy()
        counts = df_counts[class_cols].to_numpy(np.float64)

    obs_weight = counts.sum(axis=1).astype(np.float32)

    if np.any(obs_weight <= 0):
        raise ValueError("Found observation(s) with zero total count after processing score/score_list.")

    q_obs = (counts / obs_weight[:, None]).astype(np.float32)

    # Representative hard label for optional classical diagnostics
    # mode category, 0..K-1
    y_mode = np.argmax(q_obs, axis=1).astype(np.int64)
    df_obs["y_ord"] = y_mode

    # Factorize facets on the observation table actually used by Stage 2
    df_obs["item_idx"], item_categories = pd.factorize(df_obs["item_id"])
    df_obs["system_idx"], system_categories = pd.factorize(df_obs["system_id"])
    df_obs["rater_idx"], rater_categories = pd.factorize(df_obs["rater_id"])
    df_obs["dim_idx"], dim_categories = pd.factorize(df_obs["dimension"])

    meta = {
        "item_categories": item_categories,
        "system_categories": system_categories,
        "rater_categories": rater_categories,
        "dim_categories": dim_categories,
        "n_classes": n_classes,
    }

    return df_obs, meta, q_obs, obs_weight