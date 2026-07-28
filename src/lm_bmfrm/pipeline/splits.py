"""Outer-system split construction and geometry diagnostics."""
from __future__ import annotations
from itertools import combinations
import json
import numpy as np
import pandas as pd


def _resolve_outer_n_test(n_systems, test_size):
    """Resolve a float/int outer test size to an integer number of systems."""
    if isinstance(test_size, float):
        if not (0.0 < test_size < 1.0):
            raise ValueError("Float test_size must be in (0, 1).")
        n_test = int(np.floor(n_systems * test_size))
    else:
        n_test = int(test_size)
    return max(1, min(int(n_test), n_systems - 2))


def _build_system_stratification_scores(
    systems,
    full_human_theta_by_dim=None,
    dims=None,
    stratify_dimension="__mean_z__",
):
    """
    Build one full-data human-theta score per system for stratified outer splits.

    stratify_dimension options:
      - "__mean_z__" / "mean_z" / None: mean z-scored theta across all available dims.
      - a concrete dimension name, e.g. "coherence".

    The split itself is still system-level. The full-data theta is used only to
    choose balanced held-out systems across the human ability range, not for fitting
    any outer-test model.
    """
    systems = np.array([str(s) for s in systems], dtype=object)
    systems = np.array(sorted(pd.unique(systems)), dtype=object)
    dims = [str(d) for d in (dims if dims is not None else [])]

    if not full_human_theta_by_dim:
        # Fallback deterministic pseudo-score: rank by system id. This preserves
        # old behavior only when the caller chooses random; for stratified splits
        # it still creates evenly spaced system-id folds rather than crashing.
        return {str(s): float(i) for i, s in enumerate(systems)}, {
            "stratify_source": "fallback_system_id_rank",
            "stratify_dimension": str(stratify_dimension),
        }

    theta_by_dim = {
        str(dim): {str(k): float(v) for k, v in theta_map.items()}
        for dim, theta_map in full_human_theta_by_dim.items()
        if isinstance(theta_map, dict) and len(theta_map) > 0
    }

    if stratify_dimension and str(stratify_dimension) not in {"__mean_z__", "mean_z", "mean", "auto"}:
        dim = str(stratify_dimension)
        if dim not in theta_by_dim:
            raise ValueError(
                f"outer_stratify_dimension={dim!r} was requested, but no full-data "
                f"human theta map was available for it. Available dimensions: {sorted(theta_by_dim)}"
            )
        score_map = {str(s): float(theta_by_dim[dim].get(str(s), np.nan)) for s in systems}
        source = f"full_human_theta:{dim}"
    else:
        available_dims = [d for d in dims if d in theta_by_dim]
        if not available_dims:
            available_dims = sorted(theta_by_dim)
        z_blocks = []
        for dim in available_dims:
            vals = np.array([theta_by_dim[dim].get(str(s), np.nan) for s in systems], dtype=float)
            mu = np.nanmean(vals)
            sd = np.nanstd(vals)
            if not np.isfinite(sd) or sd < 1e-8:
                z = vals - mu
            else:
                z = (vals - mu) / sd
            z_blocks.append(z)
        Z = np.vstack(z_blocks) if z_blocks else np.zeros((1, len(systems)))
        scores = np.nanmean(Z, axis=0)
        score_map = {str(s): float(scores[i]) for i, s in enumerate(systems)}
        source = "full_human_theta:mean_z(" + ",".join(available_dims) + ")"

    # If some system is missing from theta maps, put it at the median so the split
    # remains possible while making the issue visible in the manifest.
    finite_vals = np.array([v for v in score_map.values() if np.isfinite(v)], dtype=float)
    fill = float(np.nanmedian(finite_vals)) if finite_vals.size else 0.0
    missing = []
    for s in systems:
        if not np.isfinite(score_map.get(str(s), np.nan)):
            score_map[str(s)] = fill
            missing.append(str(s))

    meta = {
        "stratify_source": source,
        "stratify_dimension": str(stratify_dimension),
        "n_stratify_missing_systems": len(missing),
        "stratify_missing_systems_json": json.dumps(missing),
    }
    return score_map, meta


def _resolve_outer_stratify_dimension_for_dim(requested, dim):
    """Resolve per-dimension stratification requests."""
    if requested is None:
        return str(dim)
    req = str(requested)
    if req in {"__per_dimension__", "per_dimension", "by_dimension", "dimension", "auto"}:
        return str(dim)
    return req


def _lookup_pair_probability(pairwise_map, a, b):
    """Return Pr(a>b) from a possibly orientation-specific pairwise map."""
    if pairwise_map is None:
        return np.nan
    a = str(a)
    b = str(b)
    if (a, b) in pairwise_map:
        return float(pairwise_map[(a, b)])
    if (b, a) in pairwise_map:
        return 1.0 - float(pairwise_map[(b, a)])
    # tolerate string-keyed maps produced by external code
    for sep in ["||", "|", ",", "__"]:
        k = f"{a}{sep}{b}"
        rk = f"{b}{sep}{a}"
        if k in pairwise_map:
            return float(pairwise_map[k])
        if rk in pairwise_map:
            return 1.0 - float(pairwise_map[rk])
    return np.nan


def _summarize_candidate_split_geometry(test_systems, theta_score_map, pairwise_map=None, tau_h=0.8):
    """Human-ruler geometry for one candidate outer test fold."""
    test_systems = [str(s) for s in test_systems]
    vals = np.array([float(theta_score_map.get(str(s), np.nan)) for s in test_systems], dtype=float)
    finite_vals = vals[np.isfinite(vals)]
    theta_spread = float(finite_vals.max() - finite_vals.min()) if finite_vals.size else np.nan

    n_pairs = 0
    n_supported = 0
    margins = []
    for i, a in enumerate(test_systems):
        for b in test_systems[i + 1:]:
            n_pairs += 1
            p = _lookup_pair_probability(pairwise_map, a, b)
            if np.isfinite(p):
                margin = abs(2.0 * float(p) - 1.0)
            else:
                # Fallback when posterior pairwise probabilities are unavailable:
                # use the empirical theta score gap only for split diagnostics.
                # NOTE: this compares a logit-scale gap against a probability-margin
                # threshold; it is a diagnostics-only fallback and should not be hit
                # when full_human_pairwise_by_dim is supplied.
                margin = abs(float(theta_score_map.get(a, 0.0)) - float(theta_score_map.get(b, 0.0)))
            margins.append(float(margin))
            if margin >= float(tau_h):
                n_supported += 1

    human_sr = float(n_supported / n_pairs) if n_pairs > 0 else np.nan
    baseline = float(np.mean(np.square(margins))) if margins else np.nan
    mean_abs_margin = float(np.mean(np.abs(margins))) if margins else np.nan
    return {
        "n_pairs": int(n_pairs),
        "n_supported_pairs": int(n_supported),
        "n_unsupported_pairs": int(n_pairs - n_supported),
        "human_sr": human_sr,
        "human_pairwise_baseline": baseline,
        "human_mean_abs_margin": mean_abs_margin,
        "theta_spread": theta_spread,
    }


def _rank_blocks_for_systems(test_systems, ranked, n_blocks):
    """Return the ability-rank block index for each held-out system."""
    ranked = [str(s) for s in ranked]
    rank_lookup = {s: i for i, s in enumerate(ranked)}
    blocks = np.array_split(np.arange(len(ranked)), int(max(1, n_blocks)))
    block_lookup = {}
    for block_id, idxs in enumerate(blocks):
        for idx in idxs:
            block_lookup[int(idx)] = int(block_id)
    return [block_lookup.get(rank_lookup[str(s)], -1) for s in test_systems]


def _make_constrained_blocked_outer_system_splits(
    systems,
    theta_score_map,
    pairwise_map=None,
    test_size=0.30,
    n_outer_splits=10,
    random_state=42,
    tau_h=0.8,
    min_supported_pairs=None,
    max_supported_pairs=None,
    min_sr_h=None,
    max_sr_h=None,
    min_spread_fraction=0.50,
    max_candidate_splits=20000,
):
    """
    Constrained ability-blocked system-level outer splits.

    Candidate held-out folds are accepted only if they contain both enough
    human-supported and human-unsupported pairwise comparisons, and if they span
    a minimum fraction of the occupied full-data human-theta range.

    The constraints depend only on human-ruler geometry, never on judge metrics.
    """
    systems = np.array([str(s) for s in systems], dtype=object)
    systems = np.array(sorted(pd.unique(systems)), dtype=object)
    n_systems = len(systems)
    if n_systems < 4:
        raise ValueError("Need at least 4 systems for a constrained outer split plus inner LOOCV.")

    n_test = _resolve_outer_n_test(n_systems, test_size)
    n_pairs = int(n_test * (n_test - 1) // 2)
    if n_pairs <= 0:
        raise ValueError("The outer test fold must contain at least two systems.")

    if min_supported_pairs is None:
        min_supported_pairs = int(np.floor(0.40 * n_pairs))
    if max_supported_pairs is None:
        max_supported_pairs = max(int(n_pairs - 1), int(min_supported_pairs))
    min_supported_pairs = int(max(0, min_supported_pairs))
    max_supported_pairs = int(min(n_pairs, max_supported_pairs))
    if min_supported_pairs > max_supported_pairs:
        raise ValueError(
            f"Invalid constrained split support bounds: min_supported_pairs={min_supported_pairs} "
            f"> max_supported_pairs={max_supported_pairs}."
        )

    ranked = sorted(
        [str(s) for s in systems],
        key=lambda s: (float(theta_score_map.get(str(s), 0.0)), str(s)),
    )
    full_vals = np.array([float(theta_score_map.get(str(s), np.nan)) for s in ranked], dtype=float)
    finite_full = full_vals[np.isfinite(full_vals)]
    full_spread = float(finite_full.max() - finite_full.min()) if finite_full.size else 0.0
    min_spread = float(min_spread_fraction) * full_spread if full_spread > 0 else 0.0

    all_combos = list(combinations(ranked, n_test))
    rng = np.random.default_rng(random_state)
    if len(all_combos) > int(max_candidate_splits):
        idx = rng.choice(len(all_combos), size=int(max_candidate_splits), replace=False)
        candidate_combos = [all_combos[int(i)] for i in idx]
    else:
        candidate_combos = all_combos

    accepted = []
    rejected_low_support = rejected_high_support = rejected_spread = rejected_sr = 0
    support_counts = []
    spread_values = []
    sr_values = []

    for combo in candidate_combos:
        combo = tuple(str(s) for s in combo)
        geom = _summarize_candidate_split_geometry(combo, theta_score_map, pairwise_map=pairwise_map, tau_h=tau_h)
        support_counts.append(geom["n_supported_pairs"])
        spread_values.append(geom["theta_spread"])
        sr_values.append(geom["human_sr"])

        if np.isfinite(geom["theta_spread"]) and geom["theta_spread"] < min_spread:
            rejected_spread += 1
            continue
        if geom["n_supported_pairs"] < min_supported_pairs:
            rejected_low_support += 1
            continue
        if geom["n_supported_pairs"] > max_supported_pairs:
            rejected_high_support += 1
            continue
        if min_sr_h is not None and np.isfinite(geom["human_sr"]) and geom["human_sr"] < float(min_sr_h):
            rejected_sr += 1
            continue
        if max_sr_h is not None and np.isfinite(geom["human_sr"]) and geom["human_sr"] > float(max_sr_h):
            rejected_sr += 1
            continue
        accepted.append((combo, geom))

    if not accepted:
        msg = (
            "No constrained outer splits were accepted. "
            f"n_systems={n_systems}, n_test={n_test}, n_pairs={n_pairs}, "
            f"min_supported_pairs={min_supported_pairs}, max_supported_pairs={max_supported_pairs}, "
            f"min_spread_fraction={min_spread_fraction}, tau_h={tau_h}. "
        )
        if support_counts:
            msg += (
                f"Observed supported-pair range among candidates: "
                f"{int(np.nanmin(support_counts))}..{int(np.nanmax(support_counts))}; "
                f"theta-spread range: {float(np.nanmin(spread_values)):.4g}..{float(np.nanmax(spread_values)):.4g}. "
                "Relax the support bounds or min_spread_fraction."
            )
        raise ValueError(msg)

    order = rng.permutation(len(accepted))
    replace = len(accepted) < int(n_outer_splits)
    if replace:
        chosen = rng.choice(len(accepted), size=int(n_outer_splits), replace=True)
    else:
        chosen = order[:int(n_outer_splits)]

    acceptance_rate = float(len(accepted) / max(1, len(candidate_combos)))
    for outer_fold, accepted_idx in enumerate(chosen):
        combo, geom = accepted[int(accepted_idx)]
        test_systems = sorted([str(s) for s in combo])
        dev_systems = sorted([s for s in ranked if s not in set(test_systems)])
        yield outer_fold, dev_systems, test_systems, {
            "test_rank_blocks_json": json.dumps(_rank_blocks_for_systems(test_systems, ranked, n_test)),
            "test_theta_scores_json": json.dumps({s: float(theta_score_map.get(s, np.nan)) for s in test_systems}, sort_keys=True),
            "dev_theta_scores_json": json.dumps({s: float(theta_score_map.get(s, np.nan)) for s in dev_systems}, sort_keys=True),
            "ranked_systems_json": json.dumps(ranked),
            "constraint_tau_h": float(tau_h),
            "constraint_min_supported_pairs": int(min_supported_pairs),
            "constraint_max_supported_pairs": int(max_supported_pairs),
            "constraint_min_sr_h": np.nan if min_sr_h is None else float(min_sr_h),
            "constraint_max_sr_h": np.nan if max_sr_h is None else float(max_sr_h),
            "constraint_min_spread_fraction": float(min_spread_fraction),
            "constraint_full_theta_spread": float(full_spread),
            "constraint_min_theta_spread": float(min_spread),
            "constraint_n_candidate_splits": int(len(candidate_combos)),
            "constraint_n_accepted_splits": int(len(accepted)),
            "constraint_acceptance_rate": acceptance_rate,
            "constraint_rejected_low_support": int(rejected_low_support),
            "constraint_rejected_high_support": int(rejected_high_support),
            "constraint_rejected_spread": int(rejected_spread),
            "constraint_rejected_sr": int(rejected_sr),
            "constraint_sampled_with_replacement": bool(replace),
            "test_n_pairs": int(geom["n_pairs"]),
            "test_n_supported_pairs": int(geom["n_supported_pairs"]),
            "test_n_unsupported_pairs": int(geom["n_unsupported_pairs"]),
            "test_human_sr": float(geom["human_sr"]),
            "test_human_pairwise_baseline": float(geom["human_pairwise_baseline"]),
            "test_human_mean_abs_margin": float(geom["human_mean_abs_margin"]),
            "test_theta_spread": float(geom["theta_spread"]),
        }


def _make_stratified_theta_outer_system_splits(
    systems,
    theta_score_map,
    test_size=0.30,
    n_outer_splits=1,
    random_state=42,
):
    """
    Stratified system-level outer splits using full-data human theta scores.

    For n=17 and test_size=0.30, n_test=floor(17*.30)=5. We sort systems by
    the full-data human theta score, divide the ordered systems into 5 rank blocks,
    and sample one test system from each block. This prevents pathological test
    folds containing only nearly tied systems, while preserving a strict outer test
    set for fitting/calibration.
    """
    systems = np.array([str(s) for s in systems], dtype=object)
    systems = np.array(sorted(pd.unique(systems)), dtype=object)
    n_systems = len(systems)
    if n_systems < 4:
        raise ValueError("Need at least 4 systems for a 70/30 outer split plus inner LOOCV.")

    n_test = _resolve_outer_n_test(n_systems, test_size)
    rng = np.random.default_rng(random_state)

    ranked = sorted(
        [str(s) for s in systems],
        key=lambda s: (float(theta_score_map.get(str(s), 0.0)), str(s)),
    )
    rank_lookup = {s: i for i, s in enumerate(ranked)}
    blocks = [list(map(str, block)) for block in np.array_split(np.array(ranked, dtype=object), n_test)]

    for outer_fold in range(int(n_outer_splits)):
        test_systems = []
        block_ids = []
        for block_id, block in enumerate(blocks):
            if len(block) == 0:
                continue
            sid = str(rng.choice(block))
            test_systems.append(sid)
            block_ids.append(block_id)

        # In the unlikely event of duplicate choices due to malformed blocks, fill
        # from the remaining systems closest to evenly spaced ranks.
        test_systems = list(dict.fromkeys(test_systems))
        if len(test_systems) < n_test:
            remaining = [s for s in ranked if s not in test_systems]
            target_ranks = np.linspace(0, n_systems - 1, n_test)
            while len(test_systems) < n_test and remaining:
                used_ranks = {rank_lookup[s] for s in test_systems}
                candidates = [s for s in remaining if rank_lookup[s] not in used_ranks] or remaining
                target = target_ranks[len(test_systems)]
                sid = min(candidates, key=lambda s: abs(rank_lookup[s] - target))
                test_systems.append(sid)
                remaining.remove(sid)

        test_systems = sorted(test_systems)
        dev_systems = sorted([s for s in ranked if s not in set(test_systems)])
        yield outer_fold, dev_systems, test_systems, {
            "test_rank_blocks_json": json.dumps(block_ids),
            "test_theta_scores_json": json.dumps({s: float(theta_score_map.get(s, np.nan)) for s in test_systems}, sort_keys=True),
            "dev_theta_scores_json": json.dumps({s: float(theta_score_map.get(s, np.nan)) for s in dev_systems}, sort_keys=True),
            "ranked_systems_json": json.dumps(ranked),
        }


def _make_outer_system_splits(
    systems,
    test_size=0.30,
    n_outer_splits=1,
    random_state=42,
    split_strategy="random",
    full_human_theta_by_dim=None,
    full_human_pairwise_by_dim=None,
    dims=None,
    stratify_dimension="__per_dimension__",
    current_dimension=None,
    constrained_tau_h=0.8,
    constrained_min_supported_pairs=None,
    constrained_max_supported_pairs=None,
    constrained_min_sr_h=None,
    constrained_max_sr_h=None,
    constrained_min_spread_fraction=0.50,
    constrained_max_candidate_splits=20000,
):
    """
    System-level outer splits.

    split_strategy="random" preserves the previous behavior.
    split_strategy="stratified_theta" uses Stage-1 full-data human theta to hold
    out systems across the human-ruler ability range.
    split_strategy="constrained_blocked" additionally requires the outer test
    fold to contain both human-supported and human-unsupported pairwise decisions.
    """
    systems = np.array([str(s) for s in systems], dtype=object)
    systems = np.array(sorted(pd.unique(systems)), dtype=object)
    n_systems = len(systems)
    if n_systems < 4:
        raise ValueError("Need at least 4 systems for a 70/30 outer split plus inner LOOCV.")

    split_strategy = str(split_strategy or "random")
    dim_for_split = _resolve_outer_stratify_dimension_for_dim(
        stratify_dimension,
        current_dimension if current_dimension is not None else (dims[0] if dims else None),
    )

    if split_strategy in {"stratified", "stratified_theta", "theta", "human_theta", "constrained", "constrained_blocked", "blocked_constrained"}:
        score_map, score_meta = _build_system_stratification_scores(
            systems=systems,
            full_human_theta_by_dim=full_human_theta_by_dim,
            dims=dims,
            stratify_dimension=dim_for_split,
        )

        if split_strategy in {"constrained", "constrained_blocked", "blocked_constrained"}:
            pairwise_map = None
            if isinstance(full_human_pairwise_by_dim, dict):
                pairwise_map = full_human_pairwise_by_dim.get(str(dim_for_split))
            for outer_fold, dev_systems, test_systems, split_meta in _make_constrained_blocked_outer_system_splits(
                systems=systems,
                theta_score_map=score_map,
                pairwise_map=pairwise_map,
                test_size=test_size,
                n_outer_splits=n_outer_splits,
                random_state=random_state,
                tau_h=constrained_tau_h,
                min_supported_pairs=constrained_min_supported_pairs,
                max_supported_pairs=constrained_max_supported_pairs,
                min_sr_h=constrained_min_sr_h,
                max_sr_h=constrained_max_sr_h,
                min_spread_fraction=constrained_min_spread_fraction,
                max_candidate_splits=constrained_max_candidate_splits,
            ):
                meta = {
                    "outer_split_strategy": "constrained_blocked",
                    "outer_split_dimension": str(current_dimension) if current_dimension is not None else str(dim_for_split),
                    **score_meta,
                    **split_meta,
                }
                yield outer_fold, dev_systems, test_systems, meta
            return

        for outer_fold, dev_systems, test_systems, split_meta in _make_stratified_theta_outer_system_splits(
            systems=systems,
            theta_score_map=score_map,
            test_size=test_size,
            n_outer_splits=n_outer_splits,
            random_state=random_state,
        ):
            meta = {
                "outer_split_strategy": "stratified_theta",
                "outer_split_dimension": str(current_dimension) if current_dimension is not None else str(dim_for_split),
                **score_meta,
                **split_meta,
            }
            yield outer_fold, dev_systems, test_systems, meta
        return

    if split_strategy != "random":
        raise ValueError("outer_split_strategy must be one of: random, stratified_theta, constrained_blocked.")

    n_test = _resolve_outer_n_test(n_systems, test_size)
    rng = np.random.default_rng(random_state)
    for outer_fold in range(int(n_outer_splits)):
        perm = rng.permutation(systems)
        test_systems = sorted([str(x) for x in perm[:n_test]])
        dev_systems = sorted([str(x) for x in perm[n_test:]])
        yield outer_fold, dev_systems, test_systems, {
            "outer_split_strategy": "random",
            "outer_split_dimension": str(current_dimension) if current_dimension is not None else "all",
        }
