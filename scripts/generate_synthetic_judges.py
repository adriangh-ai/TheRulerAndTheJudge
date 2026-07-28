#!/usr/bin/env python
"""
Generate a synthetic judge suite for LLM-as-a-judge meta-evaluation.


Typical usage
-------------

python generate_synthetic_judges.py \
    --data summeval \
    --dimension coherence \
    --target-raw-spearman 0.90 \
    --suite main \
    --output synthetic_judges_reviewer_main.parquet

python generate_synthetic_judges.py \
    --data summeval \
    --dimension coherence \
    --target-raw-spearman 0.90 \
    --suite full \
    --output synthetic_judges_reviewer_full.parquet

Input DataFrame must contain:
    rater_id, dimension, system_id, item_id, score

Output DataFrame contains those columns plus:
    pathology, pathology_family, expected_diagnostic_channel,
    pathology_description, pathology_params, paper_table_role

The resulting parquet can be used like your previous synthetic_judges.parquet.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from itertools import combinations
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

try:
    from src.lm_bmfrm.data_preprocess import (
        load_all_human_judgments,
        load_hanna,
        load_hitz_basse,
    )
except Exception:  # pragma: no cover - only used when running outside project env
    load_all_human_judgments = None
    load_hanna = None
    load_hitz_basse = None


# -----------------------------------------------------------------------------
# Global constants
# -----------------------------------------------------------------------------

SCORE_MIN = 1
SCORE_MAX = 5
SCORE_VALUES = np.arange(SCORE_MIN, SCORE_MAX + 1)


# -----------------------------------------------------------------------------
# Basic utilities
# -----------------------------------------------------------------------------

def _clip_and_round(x, min_val: int = SCORE_MIN, max_val: int = SCORE_MAX) -> np.ndarray:
    return np.clip(np.rint(np.asarray(x, dtype=float)), min_val, max_val).astype(int)


def _safe_name(x: str) -> str:
    return str(x).replace("/", "__").replace(" ", "_").replace("%", "pct").replace(".", "p")


def _jsonable(x):
    """Small helper for JSON metadata with numpy/pandas scalars."""
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, (np.ndarray,)):
        return [_jsonable(v) for v in x.tolist()]
    if isinstance(x, (pd.Series, pd.Index)):
        return [_jsonable(v) for v in x.tolist()]
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    return x


def _entropy_from_counts(counts: np.ndarray) -> float:
    counts = np.asarray(counts, dtype=float)
    if counts.sum() <= 0:
        return float("nan")
    p = counts / counts.sum()
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())


def _normalized_entropy_from_scores(scores: pd.Series) -> float:
    counts = scores.value_counts().reindex(SCORE_VALUES, fill_value=0).values
    h = _entropy_from_counts(counts)
    return float(h / np.log(len(SCORE_VALUES)))


def _category_tv(p_scores: pd.Series, q_scores: pd.Series) -> float:
    p = p_scores.value_counts(normalize=True).reindex(SCORE_VALUES, fill_value=0).values
    q = q_scores.value_counts(normalize=True).reindex(SCORE_VALUES, fill_value=0).values
    return float(0.5 * np.abs(p - q).sum())


def _system_means(df: pd.DataFrame, score_col: str = "score") -> pd.Series:
    return df.groupby("system_id")[score_col].mean().sort_index()


def _system_item_means(df: pd.DataFrame, score_col: str = "score") -> pd.Series:
    return df.groupby(["system_id", "item_id"])[score_col].mean().sort_index()


def _raw_system_spearman(synth_df: pd.DataFrame, human_system_means: pd.Series) -> float:
    synth_means = _system_means(synth_df)
    common = sorted(set(synth_means.index) & set(human_system_means.index))
    if len(common) < 3:
        return float("nan")
    rho = spearmanr(
        [synth_means.loc[s] for s in common],
        [human_system_means.loc[s] for s in common],
    ).correlation
    return float(rho) if rho is not None else float("nan")


def _raw_item_spearman(synth_df: pd.DataFrame, human_si_means: pd.Series) -> float:
    synth_si = _system_item_means(synth_df)
    common = synth_si.index.intersection(human_si_means.index)
    if len(common) < 3:
        return float("nan")
    rho = spearmanr(synth_si.loc[common].values, human_si_means.loc[common].values).correlation
    return float(rho) if rho is not None else float("nan")


def _mean_abs_score_change(a: pd.DataFrame, b: pd.DataFrame) -> float:
    keys = ["system_id", "item_id"]
    aa = a.sort_values(keys)["score"].astype(float).values
    bb = b.sort_values(keys)["score"].astype(float).values
    if len(aa) != len(bb):
        return float("nan")
    return float(np.mean(np.abs(aa - bb)))


def _frac_score_changed(a: pd.DataFrame, b: pd.DataFrame) -> float:
    keys = ["system_id", "item_id"]
    aa = a.sort_values(keys)["score"].astype(int).values
    bb = b.sort_values(keys)["score"].astype(int).values
    if len(aa) != len(bb):
        return float("nan")
    return float(np.mean(aa != bb))


# -----------------------------------------------------------------------------
# Data loading
# -----------------------------------------------------------------------------

def _load_data(data_arg: str) -> pd.DataFrame:
    if data_arg == "summeval":
        if load_all_human_judgments is None:
            raise RuntimeError("Project loader unavailable: load_all_human_judgments")
        return load_all_human_judgments(use_summeval_hf=True)

    if data_arg == "hanna":
        if load_hanna is None:
            raise RuntimeError("Project loader unavailable: load_hanna")
        return load_hanna()

    if data_arg == "hitz_basse":
        if load_hitz_basse is None:
            raise RuntimeError("Project loader unavailable: load_hitz_basse")
        return load_hitz_basse()

    if data_arg == "hitz_basse_vas":
        if load_hitz_basse is None:
            raise RuntimeError("Project loader unavailable: load_hitz_basse")
        return load_hitz_basse(split="eu")

    if data_arg.endswith(".parquet"):
        return pd.read_parquet(data_arg)

    if data_arg.endswith(".csv"):
        return pd.read_csv(data_arg)

    raise ValueError(f"Unknown --data value/path: {data_arg}")


def _filter_human_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Best-effort filtering: project loaders usually already return human rows."""
    out = df.copy()

    if "is_llm" in out.columns:
        out = out[~out["is_llm"].astype(bool)].copy()

    # In some project files, LLM raters contain a slash (provider/model).
    if "rater_id" in out.columns:
        no_slash = out[~out["rater_id"].astype(str).str.contains("/", na=False)].copy()
        # Avoid accidentally dropping all data when human IDs contain slashes.
        if len(no_slash) >= max(1, int(0.5 * len(out))):
            out = no_slash

    return out


# -----------------------------------------------------------------------------
# Empirical aggregate identity
# -----------------------------------------------------------------------------

def _make_aggregate_identity(
    human_dim: pd.DataFrame,
    mode: str = "round_mean",
    seed: int = 42,
) -> pd.DataFrame:
    """
    Create a one-rating-per-system-item empirical identity judge.

    mode = round_mean:
        y_id = round(mean human score for system-item)
    mode = sample_empirical:
        y_id sampled from empirical human labels for the system-item
    mode = median:
        y_id = round(median human score for system-item)
    """
    rng = np.random.default_rng(seed)
    rows = []

    for (sid, iid), g in human_dim.groupby(["system_id", "item_id"]):
        scores = g["score"].astype(int).values

        if mode == "round_mean":
            score = int(_clip_and_round([scores.mean()])[0])
        elif mode == "median":
            score = int(_clip_and_round([np.median(scores)])[0])
        elif mode == "sample_empirical":
            score = int(rng.choice(scores))
        else:
            raise ValueError(f"Unknown identity mode: {mode}")

        rows.append({"system_id": sid, "item_id": iid, "score": score})

    out = pd.DataFrame(rows)
    out["dimension"] = human_dim["dimension"].iloc[0]
    return out[["system_id", "item_id", "dimension", "score"]]


# -----------------------------------------------------------------------------
# Metadata attachment
# -----------------------------------------------------------------------------

def _attach_meta(
    df: pd.DataFrame,
    pathology: str,
    family: str,
    channel: str,
    description: str,
    params: Dict,
    role: str,
    prefix: str = "synthetic",
) -> pd.DataFrame:
    out = df.copy()
    out["score"] = out["score"].astype(int)
    out["rater_id"] = f"{prefix}__{pathology}"
    out["pathology"] = pathology
    out["pathology_family"] = family
    out["expected_diagnostic_channel"] = channel
    out["pathology_description"] = description
    out["pathology_params"] = json.dumps(_jsonable(params), sort_keys=True)
    out["paper_table_role"] = role
    return out


# -----------------------------------------------------------------------------
# Score transport helpers
# -----------------------------------------------------------------------------

def discrete_gaussian_probs(center, sigma=0.45, values=SCORE_VALUES):
    """
    Discrete Gaussian over ordinal categories 1..5.
    """
    values = np.asarray(values, dtype=float)
    z = np.exp(-0.5 * ((values - center) / sigma) ** 2)
    z = np.clip(z, 1e-12, None)
    return z / z.sum()


def ordinal_distribution_shift(
    df,
    delta_by_system,
    sigma=0.45,
    apply_prob=1.0,
    seed=42,
):
    """
    Shift each score by a system-specific ordinal amount Delta_s using a discrete
    Gaussian transition kernel.

    This preserves raw 1--5 score outputs while inducing controlled system-level
    distribution shifts.
    """
    out = df.copy()
    rng = np.random.default_rng(seed)
    new_scores = []

    for _, row in out.iterrows():
        old_score = int(row["score"])
        sid = row["system_id"]
        delta = float(delta_by_system.get(sid, 0.0))

        if rng.uniform() > apply_prob:
            new_scores.append(old_score)
            continue

        center = old_score + delta
        probs = discrete_gaussian_probs(center=center, sigma=sigma)
        new_score = int(rng.choice(SCORE_VALUES, p=probs))
        new_scores.append(new_score)

    out["score"] = new_scores
    return out


# -----------------------------------------------------------------------------
# Basic empirical raw-score perturbations
# -----------------------------------------------------------------------------

def identity(df: pd.DataFrame, **kwargs) -> pd.DataFrame:
    return df.copy()


def scale_shift(df: pd.DataFrame, shift: int = 1, **kwargs) -> pd.DataFrame:
    out = df.copy()
    out["score"] = _clip_and_round(out["score"].values + shift)
    return out


def category_collapse_floor(df: pd.DataFrame, floor_val: int = 4, **kwargs) -> pd.DataFrame:
    out = df.copy()
    out["score"] = np.maximum(out["score"].astype(int).values, floor_val).astype(int)
    return out


def category_collapse_3level(df: pd.DataFrame, **kwargs) -> pd.DataFrame:
    """
    Monotone collapse to three effective categories:
        1/2 -> 2, 3 -> 3, 4/5 -> 4
    """
    out = df.copy()
    score = out["score"].astype(int).values
    collapsed = np.where(score <= 2, 2, np.where(score >= 4, 4, 3))
    out["score"] = collapsed.astype(int)
    return out


def item_permute_within_system(df: pd.DataFrame, seed: int = 42, **kwargs) -> pd.DataFrame:
    """
    Preserve each system's score multiset and mean exactly, but assign scores to
    different items. System-level raw Spearman is therefore unchanged, while
    item-level coherence is corrupted.
    """
    out = df.copy()
    rng = np.random.default_rng(seed)

    for sid, idx in out.groupby("system_id").groups.items():
        idx = np.asarray(list(idx))
        scores = out.loc[idx, "score"].values.copy()
        rng.shuffle(scores)
        out.loc[idx, "score"] = scores

    return out


def item_reverse_by_global_item_difficulty(
    df: pd.DataFrame,
    human_dim: pd.DataFrame,
    **kwargs,
) -> pd.DataFrame:
    """
    Adversarial item permutation: for each system, assign its high scores to
    globally low-scoring items and vice versa. Preserves each system's score
    multiset exactly.
    """
    out = df.copy()
    item_order = human_dim.groupby("item_id")["score"].mean().sort_values().index.tolist()
    item_rank = {iid: r for r, iid in enumerate(item_order)}

    for sid, g in out.groupby("system_id"):
        idx_sorted_items = (
            g.assign(_rank=g["item_id"].map(item_rank))
            .sort_values("_rank")
            .index.to_numpy()
        )
        scores_desc = np.sort(g["score"].astype(int).values)[::-1]
        out.loc[idx_sorted_items, "score"] = scores_desc

    return out


def random_noise(df: pd.DataFrame, sigma: float = 0.8, seed: int = 42, **kwargs) -> pd.DataFrame:
    out = df.copy()
    rng = np.random.default_rng(seed)
    out["score"] = _clip_and_round(out["score"].astype(float).values + rng.normal(0.0, sigma, size=len(out)))
    return out


def add_equalizer_noise(df: pd.DataFrame, sigma: float = 0.0, seed: int = 0, **kwargs) -> pd.DataFrame:
    """
    Add ordinary score noise after a structured pathology.

    This is used only for matched-raw-Spearman counterexamples. Some structured
    pathologies preserve the system rank almost perfectly, so raw Spearman stays
    near 1 regardless of pathology strength. The equalizer noise lowers raw
    Spearman to the requested band without removing the structured failure mode.
    """
    if sigma <= 0:
        return df.copy()
    return random_noise(df, sigma=sigma, seed=seed)


# -----------------------------------------------------------------------------
# Distributional pairwise-geometry perturbations
# -----------------------------------------------------------------------------

def distributional_compression(
    df,
    lam=0.5,
    sigma=0.45,
    apply_prob=1.0,
    seed=42,
    **kwargs,
):
    """
    System-level distributional compression:
        mu'_s = grand + lam * (mu_s - grand)

    lam < 1 compresses systems toward the grand mean.
    """
    means = _system_means(df)
    grand = float(means.mean())
    target_means = grand + lam * (means - grand)
    delta_by_system = (target_means - means).to_dict()

    return ordinal_distribution_shift(
        df,
        delta_by_system=delta_by_system,
        sigma=sigma,
        apply_prob=apply_prob,
        seed=seed,
    )


def distributional_overseparation(
    df,
    alpha=2.0,
    sigma=0.45,
    apply_prob=1.0,
    seed=42,
    **kwargs,
):
    """
    System-level distributional over-separation:
        mu'_s = grand + alpha * (mu_s - grand)

    alpha > 1 pushes systems away from the grand mean.
    """
    means = _system_means(df)
    grand = float(means.mean())
    target_means = grand + alpha * (means - grand)
    delta_by_system = (target_means - means).to_dict()

    return ordinal_distribution_shift(
        df,
        delta_by_system=delta_by_system,
        sigma=sigma,
        apply_prob=apply_prob,
        seed=seed,
    )


def _target_means_false_confidence_close(
    means,
    gap_threshold=0.15,
    target_gap=0.40,
):
    """
    Increase small adjacent gaps in system mean space.
    """
    sorted_means = means.sort_values()
    systems = sorted_means.index.to_list()
    vals = sorted_means.to_numpy(dtype=float)

    gaps = np.diff(vals)
    new_gaps = np.where(gaps < gap_threshold, np.maximum(gaps, target_gap), gaps)

    new_vals = [0.0]
    for g in new_gaps:
        new_vals.append(new_vals[-1] + float(g))
    new_vals = np.asarray(new_vals)

    # Keep same center.
    new_vals = new_vals - new_vals.mean() + vals.mean()
    return pd.Series(new_vals, index=systems).reindex(means.index)


def distributional_false_confidence_close(
    df,
    gap_threshold=0.15,
    target_gap=0.40,
    sigma=0.45,
    apply_prob=1.0,
    seed=42,
    **kwargs,
):
    """
    Push close systems apart at the system-distribution level.
    """
    means = _system_means(df)
    target_means = _target_means_false_confidence_close(
        means,
        gap_threshold=gap_threshold,
        target_gap=target_gap,
    )
    delta_by_system = (target_means - means).to_dict()

    return ordinal_distribution_shift(
        df,
        delta_by_system=delta_by_system,
        sigma=sigma,
        apply_prob=apply_prob,
        seed=seed,
    )


def _target_means_miss_separated(
    means,
    gap_threshold=0.40,
    target_gap=0.15,
):
    """
    Shrink large adjacent gaps in system mean space.
    """
    sorted_means = means.sort_values()
    systems = sorted_means.index.to_list()
    vals = sorted_means.to_numpy(dtype=float)

    gaps = np.diff(vals)
    new_gaps = np.where(gaps > gap_threshold, np.minimum(gaps, target_gap), gaps)

    new_vals = [0.0]
    for g in new_gaps:
        new_vals.append(new_vals[-1] + float(g))
    new_vals = np.asarray(new_vals)

    new_vals = new_vals - new_vals.mean() + vals.mean()
    return pd.Series(new_vals, index=systems).reindex(means.index)


def distributional_miss_separated(
    df,
    gap_threshold=0.40,
    target_gap=0.15,
    sigma=0.45,
    apply_prob=1.0,
    seed=42,
    **kwargs,
):
    """
    Compress large human-separated gaps at the score-distribution level.
    """
    means = _system_means(df)
    target_means = _target_means_miss_separated(
        means,
        gap_threshold=gap_threshold,
        target_gap=target_gap,
    )
    delta_by_system = (target_means - means).to_dict()

    return ordinal_distribution_shift(
        df,
        delta_by_system=delta_by_system,
        sigma=sigma,
        apply_prob=apply_prob,
        seed=seed,
    )


# -----------------------------------------------------------------------------
# Direction-error perturbation: targeted supported-pair flip
# -----------------------------------------------------------------------------

def _select_supported_pair_proxies_from_human_means(
    human_dim: pd.DataFrame,
    available_systems: Sequence,
    top_frac: float = 0.20,
    max_pairs: int = 1,
    central_weight: bool = True,
    disjoint_systems: bool = True,
    seed: int = 42,
) -> List[Tuple[str, str, float]]:
    """
    Select high-confidence pair proxies from human aggregate system means.

    Returns tuples:
        (lower_system, higher_system, gap)

    This is a proxy for Stage-1 posterior supported pairs. It is intentionally
    simple and raw-score based so the generator stays independent of the IRT fit.
    """
    rng = np.random.default_rng(seed)
    means_all = _system_means(human_dim)
    means = means_all.reindex([s for s in available_systems if s in means_all.index]).dropna()

    systems = list(means.index)
    if len(systems) < 2:
        return []

    vals = means.to_dict()
    gaps = []
    grand = float(means.mean())
    value_range = float(means.max() - means.min())
    denom = max(value_range / 2.0, 1e-8)

    for a, b in combinations(systems, 2):
        ma, mb = float(vals[a]), float(vals[b])
        if np.isclose(ma, mb):
            continue

        low, high = (a, b) if ma < mb else (b, a)
        gap = abs(mb - ma)
        mid = 0.5 * (ma + mb)

        if central_weight:
            # Prefer supported-but-not-extreme pairs so raw Spearman is not
            # destroyed by flipping only the absolute best/worst systems.
            weight = 1.0 / (1.0 + abs(mid - grand) / denom)
            priority = gap * weight
        else:
            priority = gap

        # Add tiny random jitter for deterministic tie-breaking under seed.
        priority = priority + 1e-9 * float(rng.normal())
        gaps.append({
            "low": low,
            "high": high,
            "gap": gap,
            "priority": priority,
        })

    if not gaps:
        return []

    gap_values = np.asarray([g["gap"] for g in gaps], dtype=float)
    q = float(np.clip(1.0 - top_frac, 0.0, 1.0))
    threshold = float(np.quantile(gap_values, q))

    candidates = [g for g in gaps if g["gap"] >= threshold]
    candidates = sorted(candidates, key=lambda x: x["priority"], reverse=True)

    selected = []
    used = set()
    for c in candidates:
        if disjoint_systems and (c["low"] in used or c["high"] in used):
            continue
        selected.append((c["low"], c["high"], float(c["gap"])))
        used.add(c["low"])
        used.add(c["high"])
        if len(selected) >= max_pairs:
            break

    return selected


def targeted_supported_pair_flip(
    df: pd.DataFrame,
    human_dim: pd.DataFrame,
    top_frac: float = 0.20,
    max_pairs: int = 1,
    swap_fraction: float = 1.0,
    central_weight: bool = True,
    disjoint_systems: bool = True,
    seed: int = 42,
    **kwargs,
) -> pd.DataFrame:
    """
    Create a controlled direction error by flipping item-aligned score patterns
    between one or more human-supported pair proxies.

    Because this generator does not require a Stage-1 posterior fit, "support" is
    approximated by large human aggregate system-mean gaps. If you want the exact
    paper version, replace the pair selector with pairs satisfying:
        |2 P(theta_a > theta_b) - 1| >= tau_H

    The operation preserves the global score histogram and the combined score
    multiset of each flipped pair, but reverses the selected pair's direction.
    """
    out = df.copy()
    selected_pairs = _select_supported_pair_proxies_from_human_means(
        human_dim=human_dim,
        available_systems=out["system_id"].unique(),
        top_frac=top_frac,
        max_pairs=max_pairs,
        central_weight=central_weight,
        disjoint_systems=disjoint_systems,
        seed=seed,
    )

    rng = np.random.default_rng(seed)

    for low_system, high_system, gap in selected_pairs:
        g_low = out[out["system_id"] == low_system].set_index("item_id")["score"]
        g_high = out[out["system_id"] == high_system].set_index("item_id")["score"]
        common_items = sorted(set(g_low.index) & set(g_high.index))

        if not common_items:
            continue

        # If swap_fraction < 1, swap the items most responsible for the pair gap.
        if swap_fraction < 1.0:
            diffs = []
            for iid in common_items:
                diffs.append((iid, abs(float(g_high.loc[iid]) - float(g_low.loc[iid]))))
            diffs = sorted(diffs, key=lambda x: x[1], reverse=True)
            n_take = max(1, int(round(len(diffs) * swap_fraction)))
            common_items = [iid for iid, _ in diffs[:n_take]]
            rng.shuffle(common_items)

        for iid in common_items:
            idx_low = out[(out["system_id"] == low_system) & (out["item_id"] == iid)].index
            idx_high = out[(out["system_id"] == high_system) & (out["item_id"] == iid)].index

            if len(idx_low) and len(idx_high):
                v_low = out.loc[idx_low[0], "score"]
                v_high = out.loc[idx_high[0], "score"]
                out.loc[idx_low[0], "score"] = v_high
                out.loc[idx_high[0], "score"] = v_low

    out.attrs["synthetic_params_extra"] = {
        "selected_pair_proxy_source": "human_aggregate_system_mean_gap",
        "selected_pairs_low_high_gap": [
            {"low": str(low), "high": str(high), "gap": float(gap)}
            for low, high, gap in selected_pairs
        ],
        "n_selected_pairs": len(selected_pairs),
    }
    return out


def local_adjacent_flip(
    df: pd.DataFrame,
    which: str = "middle_largest_gap",
    **kwargs,
) -> pd.DataFrame:
    """
    Older, weaker direction-error control. Kept as an appendix-only comparator.
    """
    out = df.copy()
    means = _system_means(out).sort_values()
    systems = means.index.tolist()
    if len(systems) < 2:
        return out

    if which == "middle":
        i = max(0, min(len(systems) - 2, len(systems) // 2 - 1))
    elif which == "middle_largest_gap":
        gaps = np.diff(means.values)
        mid = (len(gaps) - 1) / 2.0
        weights = 1.0 / (1.0 + np.abs(np.arange(len(gaps)) - mid))
        i = int(np.argmax(gaps * weights))
    else:
        i = int(which)
        i = max(0, min(len(systems) - 2, i))

    s1, s2 = systems[i], systems[i + 1]
    g1 = out[out["system_id"] == s1].set_index("item_id")["score"]
    g2 = out[out["system_id"] == s2].set_index("item_id")["score"]
    common = sorted(set(g1.index) & set(g2.index))

    for iid in common:
        idx1 = out[(out["system_id"] == s1) & (out["item_id"] == iid)].index
        idx2 = out[(out["system_id"] == s2) & (out["item_id"] == iid)].index
        if len(idx1) and len(idx2):
            v1 = out.loc[idx1[0], "score"]
            v2 = out.loc[idx2[0], "score"]
            out.loc[idx1[0], "score"] = v2
            out.loc[idx2[0], "score"] = v1

    return out


# -----------------------------------------------------------------------------
# Optional latent-like raw generation; appendix only
# -----------------------------------------------------------------------------

def empirical_rank_logit_generate(
    df: pd.DataFrame,
    human_dim: pd.DataFrame,
    transform: str = "identity",
    param: float = 1.0,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate raw 1--5 scores from an ordinal-logit-like model using empirical
    system and item effects estimated from human scores.

    This is not the PyMC model; it is a lightweight raw-score-facing construct
    generator. Use it only as appendix material. The "identity" version is not an
    observable score identity; it is a latent-resampling control.
    """
    from scipy.special import expit

    rng = np.random.default_rng(seed)
    out = df.copy()

    sys_mean = human_dim.groupby("system_id")["score"].mean()
    item_mean = human_dim.groupby("item_id")["score"].mean()
    grand = float(human_dim["score"].mean())

    sys_eff = sys_mean - grand
    item_diff = grand - item_mean  # harder item => lower mean => positive difficulty

    z = sys_eff.copy()
    if transform == "compression":
        z = param * z
    elif transform == "overseparation":
        z = param * z
    elif transform == "nonlinear":
        gamma = param
        zz = z.values
        transformed = np.sign(zz) * np.power(np.abs(zz), gamma)
        if np.max(np.abs(transformed)) > 1e-12:
            transformed = transformed * (np.max(np.abs(zz)) / np.max(np.abs(transformed)))
        z = pd.Series(transformed, index=z.index)
    elif transform == "identity":
        pass
    else:
        raise ValueError(transform)

    cutpoints = np.array([-1.5, -0.5, 0.5, 1.5], dtype=float)

    scores = []
    for _, row in out.iterrows():
        eta = float(z.get(row["system_id"], 0.0) - item_diff.get(row["item_id"], 0.0))
        cdf = expit(cutpoints - eta)
        probs = np.concatenate([[cdf[0]], np.diff(cdf), [1.0 - cdf[-1]]])
        probs = np.clip(probs, 1e-8, 1.0)
        probs = probs / probs.sum()
        scores.append(int(rng.choice(SCORE_VALUES, p=probs)))

    out["score"] = scores
    return out


# -----------------------------------------------------------------------------
# Suite specification and tuning
# -----------------------------------------------------------------------------

@dataclass
class JudgeSpec:
    pathology: str
    family: str
    channel: str
    description: str
    func: Callable[..., pd.DataFrame]
    params: Dict
    role: str = "main"
    tunable_param: Optional[str] = None
    tuning_grid: Optional[Sequence[float]] = None


MAIN_FIXED_SPECS: List[JudgeSpec] = [
    JudgeSpec(
        "00_empirical_identity",
        "baseline",
        "all",
        "Empirical aggregate identity: rounded mean human score per system-item.",
        identity,
        {},
        role="main",
    ),
    JudgeSpec(
        "10_coherent_lenient_shift_plus1",
        "scale_use",
        "stage2_cutpoints",
        "Coherent leniency shift. Should be absorbed mostly as cutpoint/severity shift, not treated as a pairwise-fidelity failure.",
        scale_shift,
        {"shift": 1},
        role="main",
    ),
    JudgeSpec(
        "13_category_collapse_3level",
        "category_collapse",
        "category_entropy_emd",
        "Monotone collapse to three categories. Rank can survive; category-use diagnostics should flag reduced ordinal resolution.",
        category_collapse_3level,
        {},
        role="main",
    ),
    JudgeSpec(
        "20_item_permute_within_system",
        "item_incoherence",
        "stage2_outfit_emd",
        "Preserves each system's score multiset/mean but permutes item labels. System-level geometry should survive; item coherence should fail.",
        item_permute_within_system,
        {"seed": 123},
        role="main",
    ),
    JudgeSpec(
        "55_targeted_supported_pair_flip",
        "direction_error",
        "direction_loss_supported_direction_accuracy",
        "Flips one human-supported pair proxy. Designed to validate direction loss and supported-direction accuracy.",
        targeted_supported_pair_flip,
        {
            "top_frac": 0.20,
            "max_pairs": 1,
            "swap_fraction": 1.0,
            "central_weight": True,
            "disjoint_systems": True,
            "seed": 555,
        },
        role="main",
    ),
    JudgeSpec(
        "60_random_noise_0p8",
        "noise_control",
        "broad_degradation",
        "Uniform random score noise. Control: raw and framework metrics should degrade together.",
        random_noise,
        {"sigma": 0.8, "seed": 321},
        role="main",
    ),
]


SUPPLEMENTAL_FIXED_SPECS: List[JudgeSpec] = [
    JudgeSpec(
        "11_coherent_harsh_shift_minus1",
        "scale_use",
        "stage2_cutpoints",
        "Coherent harshness shift. Should be absorbed mostly as cutpoint/severity shift.",
        scale_shift,
        {"shift": -1},
        role="appendix",
    ),
    JudgeSpec(
        "12_category_collapse_floor4",
        "category_collapse",
        "category_entropy_emd",
        "Severe monotone category collapse: all scores below 4 become 4. Rank can survive; distributional substitution should fail.",
        category_collapse_floor,
        {"floor_val": 4},
        role="appendix",
    ),
    JudgeSpec(
        "21_item_reverse_by_global_difficulty",
        "item_incoherence",
        "stage2_outfit_emd",
        "Adversarial item permutation: high scores assigned to globally difficult items and vice versa, preserving system means.",
        item_reverse_by_global_item_difficulty,
        {},
        role="appendix",
    ),
    JudgeSpec(
        "50_local_adjacent_flip_weak",
        "direction_error",
        "direction_loss_supported_direction_accuracy",
        "Older weak direction-error control: swaps one adjacent pair in the system ranking.",
        local_adjacent_flip,
        {"which": "middle_largest_gap"},
        role="appendix",
    ),
    JudgeSpec(
        "80_dist_compression_0p5",
        "distributional_pairwise_geometry",
        "nfsm_support_recall",
        "System-level ordinal-distribution compression using Gaussian category transport.",
        distributional_compression,
        {"lam": 0.5, "sigma": 0.45, "apply_prob": 1.0, "seed": 801},
        role="appendix",
    ),
    JudgeSpec(
        "81_dist_oversep_2p0",
        "distributional_pairwise_geometry",
        "nfsm_fsr_precision",
        "System-level ordinal-distribution over-separation using Gaussian category transport.",
        distributional_overseparation,
        {"alpha": 2.0, "sigma": 0.45, "apply_prob": 1.0, "seed": 802},
        role="appendix",
    ),
    JudgeSpec(
        "82_dist_false_confidence_close",
        "distributional_pairwise_support",
        "fsr_support_precision",
        "Pushes close systems apart through system-level ordinal-distribution shifts.",
        distributional_false_confidence_close,
        {"gap_threshold": 0.15, "target_gap": 0.40, "sigma": 0.45, "seed": 803},
        role="appendix",
    ),
    JudgeSpec(
        "83_dist_miss_separated",
        "distributional_pairwise_support",
        "support_recall_nfsm",
        "Shrinks large system gaps through system-level ordinal-distribution shifts.",
        distributional_miss_separated,
        {"gap_threshold": 0.40, "target_gap": 0.15, "sigma": 0.45, "seed": 804},
        role="appendix",
    ),
]


MATCHABLE_SPECS: List[JudgeSpec] = [
    JudgeSpec(
        "m_random_noise",
        "matched_raw_spearman_noise_control",
        "broad_degradation",
        "Tuned random noise control to match a target raw Spearman.",
        random_noise,
        {"seed": 999},
        role="main",
        tunable_param="sigma",
        tuning_grid=np.linspace(0.0, 2.0, 50),
    ),
    JudgeSpec(
        "m_compression",
        "matched_raw_spearman_pairwise_geometry",
        "nfsm_support_recall",
        "Tuned distributional compression to match a target raw Spearman.",
        distributional_compression,
        {"sigma": 0.45, "apply_prob": 1.0, "seed": 901},
        role="main",
        tunable_param="lam",
        tuning_grid=np.linspace(0.05, 1.0, 40),
    ),
    JudgeSpec(
        "m_overseparation",
        "matched_raw_spearman_pairwise_geometry",
        "nfsm_fsr_precision",
        "Tuned distributional over-separation to match a target raw Spearman.",
        distributional_overseparation,
        {"sigma": 0.45, "apply_prob": 1.0, "seed": 902},
        role="main",
        tunable_param="alpha",
        tuning_grid=np.linspace(1.0, 5.0, 50),
    ),
    JudgeSpec(
        "m_false_confidence_close",
        "matched_raw_spearman_pairwise_support",
        "fsr_support_precision",
        "Tuned false confidence on close systems to match a target raw Spearman.",
        distributional_false_confidence_close,
        {"gap_threshold": 0.18, "sigma": 0.45, "apply_prob": 1.0, "seed": 903},
        role="main",
        tunable_param="target_gap",
        tuning_grid=np.linspace(0.20, 1.80, 50),
    ),
    JudgeSpec(
        "m_miss_separated",
        "matched_raw_spearman_pairwise_support",
        "support_recall_nfsm",
        "Tuned missed separation to match a target raw Spearman.",
        distributional_miss_separated,
        {"gap_threshold": 0.35, "sigma": 0.45, "apply_prob": 1.0, "seed": 904},
        role="appendix",
        tunable_param="target_gap",
        tuning_grid=np.linspace(0.00, 0.30, 40),
    ),
]


LATENT_LIKE_SPECS: List[JudgeSpec] = [
    JudgeSpec(
        "70_latent_resampled_identity",
        "latent_like_construct",
        "construct_control",
        "Raw scores generated from empirical system/item effects with identity latent transform. This is not an observable identity.",
        empirical_rank_logit_generate,
        {"transform": "identity", "param": 1.0, "seed": 700},
        role="appendix",
    ),
    JudgeSpec(
        "71_latent_resampled_compression_0p5",
        "latent_like_construct",
        "nfsm_support_recall",
        "Raw scores generated from empirical system/item effects with compressed system effects.",
        empirical_rank_logit_generate,
        {"transform": "compression", "param": 0.5, "seed": 701},
        role="appendix",
    ),
    JudgeSpec(
        "72_latent_resampled_oversep_2p0",
        "latent_like_construct",
        "nfsm_fsr_precision",
        "Raw scores generated from empirical system/item effects with over-separated system effects.",
        empirical_rank_logit_generate,
        {"transform": "overseparation", "param": 2.0, "seed": 702},
        role="appendix",
    ),
]


def _run_spec(spec: JudgeSpec, base_df: pd.DataFrame, human_dim: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
    params = dict(spec.params)
    synth = spec.func(base_df, human_dim=human_dim, **params)
    extra = synth.attrs.get("synthetic_params_extra", None)
    if extra:
        params.update(extra)
    return synth, params


def _pathology_proxy_strength(
    synth: pd.DataFrame,
    base_df: pd.DataFrame,
    spec: JudgeSpec,
    human_system_means: pd.Series,
) -> float:
    """
    Pathology-specific proxy used only to break ties among candidates that match
    the same raw Spearman band. It never uses framework metrics.
    """
    synth_means = _system_means(synth)
    base_means = _system_means(base_df)
    synth_range = float(synth_means.max() - synth_means.min())
    base_range = float(base_means.max() - base_means.min())
    name = spec.pathology.lower()
    family = spec.family.lower()
    channel = spec.channel.lower()

    # Compression/missed-separation: prefer smaller between-system range.
    if "compression" in name or "miss" in name or "recall" in channel:
        return base_range - synth_range

    # Over-separation/false-confidence: prefer larger between-system range.
    if "over" in name or "false_confidence" in name or "fsr" in channel:
        return synth_range - base_range

    # Noise controls: prefer less extra noise among equally matched candidates.
    if "noise" in name or "noise" in family:
        return -_mean_abs_score_change(synth, base_df)

    # Direction errors: prefer observable score changes, but keep raw Spearman matched by outer criterion.
    if "direction" in family or "flip" in name:
        return _mean_abs_score_change(synth, base_df)

    return _mean_abs_score_change(synth, base_df)


def _tune_spec_to_raw_spearman(
    spec: JudgeSpec,
    base_df: pd.DataFrame,
    human_dim: pd.DataFrame,
    human_system_means: pd.Series,
    target_rho: float,
    rho_tolerance: float = 0.03,
    equalizer_noise_grid: Optional[Sequence[float]] = None,
    equalizer_seeds: Sequence[int] = (1001, 1002, 1003),
) -> Tuple[pd.DataFrame, Dict]:
    """
    Tune a structured pathology to a target raw-system Spearman.

    Many structured pathologies are rank-preserving by design, so tuning their
    pathology strength alone cannot lower raw Spearman into a desired band. We
    therefore optionally add an equalizer noise layer after the pathology. The
    selected candidate is the strongest pathology among candidates inside the
    target band; if no candidate is inside the band, we return the closest one.
    """
    if spec.tunable_param is None or spec.tuning_grid is None:
        raise ValueError(f"Spec {spec.pathology} is not tunable")

    if equalizer_noise_grid is None:
        equalizer_noise_grid = [0.0]

    candidates = []

    for val in spec.tuning_grid:
        params = dict(spec.params)
        params[spec.tunable_param] = float(val)
        structured = spec.func(base_df, human_dim=human_dim, **params)

        for sigma in equalizer_noise_grid:
            seeds = [0] if float(sigma) <= 0 else list(equalizer_seeds)
            for seed in seeds:
                synth = add_equalizer_noise(structured, sigma=float(sigma), seed=int(seed))
                rho = _raw_system_spearman(synth, human_system_means)
                abs_err = abs(rho - target_rho) if np.isfinite(rho) else np.inf
                strength = _pathology_proxy_strength(synth, base_df, spec, human_system_means)

                candidates.append({
                    "abs_err": float(abs_err),
                    "rho": float(rho),
                    "value": float(val),
                    "equalizer_sigma": float(sigma),
                    "equalizer_seed": int(seed),
                    "pathology_strength_proxy": float(strength),
                    "synth": synth,
                })

    if not candidates:
        raise RuntimeError(f"No candidates generated for {spec.pathology}")

    in_band = [c for c in candidates if c["abs_err"] <= rho_tolerance]
    if in_band:
        # Among matched candidates, choose the strongest pathology. Break ties by closeness to target.
        best = sorted(in_band, key=lambda c: (-c["pathology_strength_proxy"], c["abs_err"]))[0]
        matched_in_band = True
    else:
        # If the target band is unreachable, choose the closest candidate and report explicitly.
        best = sorted(candidates, key=lambda c: (c["abs_err"], -c["pathology_strength_proxy"]))[0]
        matched_in_band = False

    tuned_params = dict(spec.params)
    tuned_params[spec.tunable_param] = best["value"]
    tuned_params["target_raw_spearman"] = float(target_rho)
    tuned_params["rho_tolerance"] = float(rho_tolerance)
    tuned_params["achieved_raw_spearman"] = float(best["rho"])
    tuned_params["abs_error_to_target"] = float(best["abs_err"])
    tuned_params["matched_in_band"] = bool(matched_in_band)
    tuned_params["tunable_param"] = spec.tunable_param
    tuned_params["equalizer_sigma"] = float(best["equalizer_sigma"])
    tuned_params["equalizer_seed"] = int(best["equalizer_seed"])
    tuned_params["pathology_strength_proxy"] = float(best["pathology_strength_proxy"])

    return best["synth"], tuned_params


def _summarize_judge(
    synth: pd.DataFrame,
    human_dim: pd.DataFrame,
    human_system_means: pd.Series,
    human_si_means: pd.Series,
    base_df: Optional[pd.DataFrame] = None,
) -> Dict:
    synth_means = _system_means(synth)
    score_dist = synth["score"].value_counts(normalize=True).reindex(SCORE_VALUES, fill_value=0.0)

    row = {
        "rater_id": synth["rater_id"].iloc[0],
        "pathology": synth["pathology"].iloc[0],
        "pathology_family": synth["pathology_family"].iloc[0],
        "expected_diagnostic_channel": synth["expected_diagnostic_channel"].iloc[0],
        "paper_table_role": synth["paper_table_role"].iloc[0],
        "raw_system_spearman_vs_human": _raw_system_spearman(synth, human_system_means),
        "raw_item_spearman_vs_human": _raw_item_spearman(synth, human_si_means),
        "system_mean_range": float(synth_means.max() - synth_means.min()),
        "system_mean_std": float(synth_means.std()),
        "score_mean": float(synth["score"].mean()),
        "score_std": float(synth["score"].std()),
        "n_categories_used": int(synth["score"].nunique()),
        "category_entropy_norm": _normalized_entropy_from_scores(synth["score"]),
        "category_tv_to_human": _category_tv(synth["score"], human_dim["score"].astype(int)),
        "pathology_description": synth["pathology_description"].iloc[0],
        "pathology_params": synth["pathology_params"].iloc[0],
    }

    if base_df is not None:
        row["frac_scores_changed_vs_identity"] = _frac_score_changed(synth, base_df)
        row["mean_abs_score_change_vs_identity"] = _mean_abs_score_change(synth, base_df)

    for k in SCORE_VALUES:
        row[f"frac_score_{k}"] = float(score_dist.loc[k])

    return row


def _select_specs_for_suite(
    suite: str,
    include_latent_like: bool,
) -> List[JudgeSpec]:
    if suite == "main":
        specs = list(MAIN_FIXED_SPECS)
    elif suite == "full":
        specs = list(MAIN_FIXED_SPECS) + list(SUPPLEMENTAL_FIXED_SPECS)
    elif suite == "appendix":
        specs = list(SUPPLEMENTAL_FIXED_SPECS)
    elif suite == "matched_only":
        specs = []
    else:
        raise ValueError(f"Unknown suite: {suite}")

    if include_latent_like and suite != "matched_only":
        specs.extend(LATENT_LIKE_SPECS)

    return specs


def generate_suite(
    human_dim: pd.DataFrame,
    identity_mode: str = "round_mean",
    seed: int = 42,
    suite: str = "main",
    include_latent_like: bool = False,
    target_raw_spearman: Optional[float] = None,
    target_rho_tolerance: float = 0.03,
    equalizer_max_sigma: float = 1.5,
    equalizer_grid_size: int = 16,
    include_appendix_matched: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    base = _make_aggregate_identity(human_dim, mode=identity_mode, seed=seed)
    human_system_means = _system_means(human_dim)
    human_si_means = _system_item_means(human_dim)

    frames: List[pd.DataFrame] = []
    summary_rows: List[Dict] = []

    specs = _select_specs_for_suite(suite=suite, include_latent_like=include_latent_like)

    for spec in specs:
        synth, used_params = _run_spec(spec, base, human_dim)
        synth = _attach_meta(
            synth,
            pathology=spec.pathology,
            family=spec.family,
            channel=spec.channel,
            description=spec.description,
            params=used_params,
            role=spec.role,
        )
        frames.append(synth)
        summary_rows.append(
            _summarize_judge(
                synth,
                human_dim,
                human_system_means,
                human_si_means,
                base_df=base,
            )
        )

    if target_raw_spearman is not None:
        equalizer_noise_grid = np.linspace(0.0, float(equalizer_max_sigma), int(equalizer_grid_size))

        match_specs = []
        for spec in MATCHABLE_SPECS:
            if spec.role == "main" or include_appendix_matched or suite in {"full", "appendix", "matched_only"}:
                match_specs.append(spec)

        for spec in match_specs:
            synth, tuned_params = _tune_spec_to_raw_spearman(
                spec,
                base,
                human_dim,
                human_system_means,
                target_raw_spearman,
                rho_tolerance=target_rho_tolerance,
                equalizer_noise_grid=equalizer_noise_grid,
            )

            target_str = str(target_raw_spearman).replace(".", "p")
            pathology = f"{spec.pathology}_rho{target_str}"
            desc = spec.description + " Selected by grid search to match raw-system Spearman."

            synth = _attach_meta(
                synth,
                pathology=pathology,
                family=spec.family,
                channel=spec.channel,
                description=desc,
                params=tuned_params,
                role=spec.role,
            )
            frames.append(synth)
            summary_rows.append(
                _summarize_judge(
                    synth,
                    human_dim,
                    human_system_means,
                    human_si_means,
                    base_df=base,
                )
            )

    if not frames:
        raise RuntimeError("No synthetic judges were generated. Check --suite and --target-raw-spearman.")

    result = pd.concat(frames, ignore_index=True)
    summary = pd.DataFrame(summary_rows)

    # Reviewer-friendly ordering.
    role_order = {"main": 0, "appendix": 1}
    summary["_role_order"] = summary["paper_table_role"].map(role_order).fillna(99)
    summary = (
        summary.sort_values(["_role_order", "pathology_family", "pathology"])
        .drop(columns=["_role_order"])
        .reset_index(drop=True)
    )

    return result, summary


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate reviewer-safe diagnostic synthetic judge suite with raw-score outputs."
    )
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="summeval, hanna, hitz_basse, hitz_basse_vas, or path to parquet/csv",
    )
    parser.add_argument(
        "--dimension",
        type=str,
        required=True,
        help="Dimension to generate synthetic judges for",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="synthetic_judges_reviewer_safe.parquet",
        help="Output parquet path",
    )
    parser.add_argument(
        "--output-summary",
        type=str,
        default=None,
        help="Summary CSV path",
    )
    parser.add_argument(
        "--suite",
        type=str,
        default="main",
        choices=["main", "full", "appendix", "matched_only"],
        help=(
            "main = reviewer-facing compact suite; "
            "full = main + appendix fixed controls; "
            "appendix = appendix fixed controls only; "
            "matched_only = only matched raw-Spearman controls, requires --target-raw-spearman"
        ),
    )
    parser.add_argument(
        "--identity-mode",
        type=str,
        default="round_mean",
        choices=["round_mean", "median", "sample_empirical"],
        help="Empirical aggregate identity construction",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--target-raw-spearman",
        type=float,
        default=None,
        help="If set, add matched raw-Spearman synthetic judges near this target, e.g. 0.90",
    )
    parser.add_argument(
        "--target-rho-tolerance",
        type=float,
        default=0.03,
        help="Accepted absolute tolerance around --target-raw-spearman for matched judges",
    )
    parser.add_argument(
        "--equalizer-max-sigma",
        type=float,
        default=1.5,
        help="Maximum post-pathology score-noise sigma used to equalize raw Spearman",
    )
    parser.add_argument(
        "--equalizer-grid-size",
        type=int,
        default=16,
        help="Number of sigma values in the equalizer-noise grid",
    )
    parser.add_argument(
        "--include-appendix-matched",
        action="store_true",
        help="Also include appendix-role matched controls such as m_miss_separated in the main suite.",
    )
    parser.add_argument(
        "--include-latent-like",
        action="store_true",
        help="Include latent-resampled construct-validity judges as appendix controls.",
    )
    parser.add_argument(
        "--write-per-judge",
        action="store_true",
        help="Also write one parquet file per synthetic judge",
    )
    args = parser.parse_args()

    if args.suite == "matched_only" and args.target_raw_spearman is None:
        raise ValueError("--suite matched_only requires --target-raw-spearman")

    df_all = _load_data(args.data)
    required = {"rater_id", "dimension", "system_id", "item_id", "score"}
    missing = required - set(df_all.columns)
    if missing:
        raise ValueError(f"Input data missing required columns: {sorted(missing)}")

    df_all = _filter_human_rows(df_all)
    human_dim = df_all[df_all["dimension"] == args.dimension].copy()

    if human_dim.empty:
        raise ValueError(
            f"No human data found for dimension={args.dimension!r}. "
            f"Available dimensions: {sorted(df_all['dimension'].dropna().unique())}"
        )

    human_dim["score"] = human_dim["score"].astype(int)

    n_systems = human_dim["system_id"].nunique()
    n_items = human_dim["item_id"].nunique()
    n_raters = human_dim["rater_id"].nunique()

    print("=" * 88)
    print("Reviewer-safe synthetic judge suite")
    print("=" * 88)
    print(f"Data: {args.data}")
    print(f"Dimension: {args.dimension}")
    print(f"Suite: {args.suite}")
    print(f"Human rows: {len(human_dim)}")
    print(f"Systems: {n_systems} | Items: {n_items} | Human raters: {n_raters}")
    print("Human score distribution:")
    print(human_dim["score"].value_counts().sort_index().to_string())

    result_df, summary_df = generate_suite(
        human_dim=human_dim,
        identity_mode=args.identity_mode,
        seed=args.seed,
        suite=args.suite,
        include_latent_like=args.include_latent_like,
        target_raw_spearman=args.target_raw_spearman,
        target_rho_tolerance=args.target_rho_tolerance,
        equalizer_max_sigma=args.equalizer_max_sigma,
        equalizer_grid_size=args.equalizer_grid_size,
        include_appendix_matched=args.include_appendix_matched,
    )

    display_cols = [
        "paper_table_role",
        "pathology",
        "pathology_family",
        "expected_diagnostic_channel",
        "raw_system_spearman_vs_human",
        "raw_item_spearman_vs_human",
        "n_categories_used",
        "category_entropy_norm",
        "category_tv_to_human",
        "system_mean_range",
        "frac_scores_changed_vs_identity",
        "mean_abs_score_change_vs_identity",
    ]

    print("\n" + "=" * 88)
    print("Summary")
    print("=" * 88)
    print(summary_df[display_cols].to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    if args.target_raw_spearman is not None:
        print("\nMatched raw-Spearman judges:")
        matched = summary_df[summary_df["pathology"].str.contains("_rho", regex=False)]
        print(
            matched[
                [
                    "paper_table_role",
                    "pathology",
                    "raw_system_spearman_vs_human",
                    "pathology_params",
                ]
            ].to_string(index=False, float_format=lambda x: f"{x:.3f}")
        )

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    result_df.to_parquet(args.output, index=False)

    summary_path = args.output_summary or args.output.replace(".parquet", "_summary.csv")
    summary_df.to_csv(summary_path, index=False)

    if args.write_per_judge:
        base_no_ext = args.output[:-8] if args.output.endswith(".parquet") else args.output
        for judge_id, g in result_df.groupby("rater_id"):
            g.to_parquet(f"{base_no_ext}_{_safe_name(judge_id)}.parquet", index=False)

    print("\nSaved:")
    print(f"  Synthetic judges: {args.output}")
    print(f"  Summary:          {summary_path}")
    print(f"  Rows:             {len(result_df)}")
    print(f"  Judges:           {result_df['rater_id'].nunique()}")

    print("\nRecommended use in paper:")
    print("  1) Treat synthetic judges as unit tests for diagnostic channels, not as a single-metric contest.")
    print("  2) Use --suite main for the main paper table and --suite full for appendix robustness.")
    print("  3) With --target-raw-spearman, emphasize: similar raw Spearman, different NFSM/FSR/EMD/outfit behavior.")
    print("  4) The targeted supported-pair flip uses human mean gaps as a proxy; replace with Stage-1 posterior support if available.")


if __name__ == "__main__":
    main()
