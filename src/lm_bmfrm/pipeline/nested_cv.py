"""Nested outer-system evaluation with inner conformal calibration."""
from __future__ import annotations
from collections import defaultdict
from itertools import product
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
from lm_bmfrm.irt_utils import conformal_q
from .fitting import fit_fold_models_per_dim
from .heldout import _evaluate_heldout_systems_from_fit
from .linking import fit_robust_affine_link
from .splits import _make_outer_system_splits, _resolve_outer_stratify_dimension_for_dim


def run_nested_system_cv_conformal(
    pm_inputs,
    dims,
    theta_grid,
    outer_test_size=0.30,
    n_outer_splits=10,
    outer_random_state=42,
    outer_split_strategy="constrained_blocked",
    full_human_theta_by_dim=None,
    full_human_pairwise_by_dim=None,
    outer_stratify_dimension="__per_dimension__",
    outer_constrained_tau_h=0.8,
    outer_min_supported_pairs=None,
    outer_max_supported_pairs=None,
    outer_min_sr_h=None,
    outer_max_sr_h=None,
    outer_min_spread_fraction=0.50,
    outer_max_candidate_splits=20000,
    alpha=0.05,
    draws_cv=1200,
    chains_cv=2,
    target_accept=0.95,
    anchor_draw_selection_method="medoids",
    anchor_draw_compute_n=64,
    anchor_draw_fit_n=None,
    use_mc=True,
    add_alpha_main=False,
    sampler_seed=42,
    anchor_draw_compute_seed=42,
    anchor_draw_fit_seed=123,
):
    """
    Stage 3 nested validation with repeated outer system splits.

    Default split design is constrained ability-blocked resampling:
      1. For each dimension independently, use the full-data Stage-1 human theta
         for that same dimension as a fold-construction blocking variable.
      2. Sample/accept outer held-out folds that span the occupied human ruler and
         contain both supported and unsupported human pairwise comparisons.
      3. Run inner LOOCV only on the development systems for that dimension.
      4. Fit the final human/judge calibration model on all development systems
         and evaluate the held-out test systems.

    The full-data human anchor is used only for split design. No held-out test
    system is used to fit the outer human anchor, judge cutpoints, model selection,
    or conformal q calibration.
    """
    pm_inputs = pm_inputs.copy()
    pm_inputs["system_id"] = pm_inputs["system_id"].astype(str)
    pm_inputs["dimension"] = pm_inputs["dimension"].astype(str)
    pm_inputs["y_ord"] = pm_inputs["score"].astype(int) - 1

    dims = [str(d) for d in dims]
    systems = pm_inputs["system_id"].drop_duplicates().to_numpy()
    theta_grid = np.asarray(theta_grid, dtype=float)
    theta_grid_anchor = theta_grid.copy()

    judge_ids_by_dim = {
        DIM: sorted(pm_inputs[(pm_inputs["is_llm"]) & (pm_inputs["dimension"] == DIM)]["rater_id"].dropna().unique())
        for DIM in dims
    }

    records = []
    rhat_records = []
    split_manifest = []
    artifact_arrays = {}
    artifact_manifest = []

    # Construct outer splits separately for each dimension when requested. This is
    # important because coherence, consistency, relevance, etc. can have different
    # human rulers and different supported-pair geometry.
    outer_splits_by_dim = {}
    for dim_idx, DIM in enumerate(dims):
        dim_seed = int(outer_random_state) + 1009 * int(dim_idx)
        outer_splits_by_dim[DIM] = list(_make_outer_system_splits(
            systems=systems,
            test_size=outer_test_size,
            n_outer_splits=n_outer_splits,
            random_state=dim_seed,
            split_strategy=outer_split_strategy,
            full_human_theta_by_dim=full_human_theta_by_dim,
            full_human_pairwise_by_dim=full_human_pairwise_by_dim,
            dims=dims,
            stratify_dimension=outer_stratify_dimension,
            current_dimension=DIM,
            constrained_tau_h=outer_constrained_tau_h,
            constrained_min_supported_pairs=outer_min_supported_pairs,
            constrained_max_supported_pairs=outer_max_supported_pairs,
            constrained_min_sr_h=outer_min_sr_h,
            constrained_max_sr_h=outer_max_sr_h,
            constrained_min_spread_fraction=outer_min_spread_fraction,
            constrained_max_candidate_splits=outer_max_candidate_splits,
        ))
        if len(outer_splits_by_dim[DIM]) != int(n_outer_splits):
            raise RuntimeError(
                f"Expected {n_outer_splits} outer splits for dimension {DIM}, "
                f"got {len(outer_splits_by_dim[DIM])}."
            )

    for outer_fold in tqdm(range(int(n_outer_splits)), desc="Outer constrained-blocked system splits"):
        for DIM in tqdm(dims, desc=f"Outer {outer_fold}: dimensions"):
            outer_fold_dim, dev_systems, test_systems, outer_split_meta = outer_splits_by_dim[DIM][outer_fold]
            if int(outer_fold_dim) != int(outer_fold):
                raise RuntimeError(
                    f"Split fold mismatch for dim={DIM}: expected {outer_fold}, got {outer_fold_dim}."
                )

            dev_set = set(dev_systems)
            test_set = set(test_systems)
            dev_df = pm_inputs[pm_inputs["system_id"].isin(dev_set)].copy()
            test_df = pm_inputs[pm_inputs["system_id"].isin(test_set)].copy()
            effective_strat_dim = _resolve_outer_stratify_dimension_for_dim(outer_stratify_dimension, DIM)

            split_manifest.append({
                "outer_fold": int(outer_fold),
                "dimension": DIM,
                "dev_systems_json": json.dumps(dev_systems),
                "test_systems_json": json.dumps(test_systems),
                "n_dev_systems": len(dev_systems),
                "n_test_systems": len(test_systems),
                "outer_test_size": outer_test_size,
                "outer_random_state": outer_random_state,
                "outer_dimension_seed": int(outer_random_state) + 1009 * dims.index(DIM),
                "outer_split_strategy": outer_split_strategy,
                "outer_stratify_dimension": outer_stratify_dimension,
                "outer_stratify_dimension_effective": effective_strat_dim,
                **outer_split_meta,
            })

            final_fit = fit_fold_models_per_dim(
                train_df=dev_df,
                DIM=DIM,
                judge_ids=judge_ids_by_dim[DIM],
                draws=draws_cv,
                target_accept=target_accept,
                chains=chains_cv,
                dif_params=False,
                anchor_draw_selection_method=anchor_draw_selection_method,
                anchor_draw_compute_n=anchor_draw_compute_n,
                anchor_draw_fit_n=(
                    max(anchor_draw_compute_n, 128)
                    if anchor_draw_fit_n is None
                    else anchor_draw_fit_n
                ),
                use_mc=use_mc,
                add_alpha=add_alpha_main,
                sampler_seed=sampler_seed,
                anchor_draw_compute_seed=anchor_draw_compute_seed,
                anchor_draw_fit_seed=anchor_draw_fit_seed,
            )
            final_dev_theta_map = final_fit["human"]["fold_theta_human_map"]

            test_records, test_rhats = _evaluate_heldout_systems_from_fit(
                fit_result=final_fit,
                eval_df=test_df,
                hold_systems=test_systems,
                DIM=DIM,
                theta_grid=theta_grid,
                theta_grid_anchor=theta_grid_anchor,
                link_a=0.0,
                link_b=1.0,
                link_rmse=0.0,
                outer_fold=outer_fold,
                eval_split="outer_test",
                inner_fold=None,
                train_system_count=len(dev_systems),
                artifact_arrays=artifact_arrays,
                artifact_manifest=artifact_manifest,
            )
            records.extend(test_records)
            rhat_records.extend(test_rhats)

            for inner_fold, sid_hold in tqdm(
                list(enumerate(dev_systems)),
                total=len(dev_systems),
                desc=f"Outer {outer_fold} dim={DIM}: inner LOOCV",
            ):
                inner_train_systems = sorted([s for s in dev_systems if s != sid_hold])
                inner_train_set = set(inner_train_systems)
                inner_train_df = dev_df[dev_df["system_id"].isin(inner_train_set)].copy()
                inner_hold_df = dev_df[dev_df["system_id"] == sid_hold].copy()

                inner_fit = fit_fold_models_per_dim(
                    train_df=inner_train_df,
                    DIM=DIM,
                    judge_ids=judge_ids_by_dim[DIM],
                    draws=draws_cv,
                    target_accept=target_accept,
                    chains=chains_cv,
                    dif_params=False,
                    anchor_draw_selection_method=anchor_draw_selection_method,
                    anchor_draw_compute_n=anchor_draw_compute_n,
                    anchor_draw_fit_n=(
                    max(anchor_draw_compute_n, 128)
                    if anchor_draw_fit_n is None
                    else anchor_draw_fit_n
                ),
                    use_mc=use_mc,
                    add_alpha=add_alpha_main,
                    sampler_seed=sampler_seed,
                    anchor_draw_compute_seed=anchor_draw_compute_seed,
                    anchor_draw_fit_seed=anchor_draw_fit_seed,
                )

                fold_theta_human_map = inner_fit["human"]["fold_theta_human_map"]
                overlap_systems = sorted(
                    set(inner_train_systems)
                    & set(fold_theta_human_map.keys())
                    & set(final_dev_theta_map.keys())
                )
                if len(overlap_systems) < 2:
                    raise ValueError(
                        f"Not enough overlap systems to link outer={outer_fold}, "
                        f"inner={inner_fold}, dim={DIM}"
                    )

                theta_fold_overlap = np.array([fold_theta_human_map[s] for s in overlap_systems], dtype=float)
                theta_anchor_overlap = np.array([final_dev_theta_map[s] for s in overlap_systems], dtype=float)
                link_a, link_b, link_rmse = fit_robust_affine_link(
                    theta_fold_overlap=theta_fold_overlap,
                    theta_anchor_overlap=theta_anchor_overlap,
                )

                loo_records, loo_rhats = _evaluate_heldout_systems_from_fit(
                    fit_result=inner_fit,
                    eval_df=inner_hold_df,
                    hold_systems=[sid_hold],
                    DIM=DIM,
                    theta_grid=theta_grid,
                    theta_grid_anchor=theta_grid_anchor,
                    link_a=link_a,
                    link_b=link_b,
                    link_rmse=link_rmse,
                    outer_fold=outer_fold,
                    eval_split="inner_loocv",
                    inner_fold=inner_fold,
                    train_system_count=len(inner_train_systems),
                    artifact_arrays=artifact_arrays,
                    artifact_manifest=artifact_manifest,
                )
                records.extend(loo_records)
                rhat_records.extend(loo_rhats)

    df_rec = pd.DataFrame(records)
    rhat_df = pd.concat(rhat_records, ignore_index=True) if rhat_records else pd.DataFrame()
    split_manifest_df = pd.DataFrame(split_manifest)
    artifact_manifest_df = pd.DataFrame(artifact_manifest)

    q_raw_by_outer = defaultdict(lambda: defaultdict(dict))
    q_norm_by_outer = defaultdict(lambda: defaultdict(dict))
    if not df_rec.empty:
        inner = df_rec[df_rec["eval_split"] == "inner_loocv"]
        for outer_fold, DIM, judge in product(
            sorted(inner["outer_fold"].unique()),
            dims,
            sorted(inner["judge_model"].unique()),
        ):
            sub = inner[
                (inner["outer_fold"] == outer_fold) &
                (inner["dimension"] == DIM) &
                (inner["judge_model"] == judge)
            ]
            if len(sub) == 0:
                continue
            q_raw_by_outer[int(outer_fold)][DIM][judge] = conformal_q(sub["resid"].values, alpha=alpha)
            q_norm_by_outer[int(outer_fold)][DIM][judge] = conformal_q(sub["resid_norm"].values, alpha=alpha)

    return {
        "predictions": df_rec,
        "q_raw_by_outer": q_raw_by_outer,
        "q_norm_by_outer": q_norm_by_outer,
        "rhat_records": rhat_df,
        "split_manifest": split_manifest_df,
        "artifact_arrays": artifact_arrays,
        "artifact_manifest": artifact_manifest_df,
    }
