"""System-level cross-validation and conformal evaluation."""
from __future__ import annotations
from collections import defaultdict
from itertools import product
import json
import arviz as az
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from tqdm import tqdm
from lm_bmfrm.evaluation_metrics import evaluate_system_item_jsd_emd
from lm_bmfrm.estimate import get_system_posterior, get_system_posterior_human
from lm_bmfrm.irt_utils import conformal_q
from .cv_common import _safe_key_part, _store_array_once
from .fitting import fit_fold_models_per_dim
from .linking import extract_anchor_system_means_by_dim, fit_robust_affine_link, transform_posterior_to_anchor_grid
from .provenance import _sampler_diagnostics


def run_system_cv_conformal(
    pm_inputs,
    dims,
    idata_anchor_by_dim,
    theta_grid,
    n_splits=5,
    alpha=0.05,
    draws_cv=2000,
    chains_cv=4,
    target_accept=0.95,
    anchor_draw_selection_method="medoids",
    anchor_draw_compute_n=64,
    anchor_draw_fit_n=None,
    save_artifacts=True,
    use_mc=True,
    add_alpha_main=False,
    sampler_seed=42,
    anchor_draw_compute_seed=42,
    anchor_draw_fit_seed=123,
    kfold_random_state=42,
):
    systems = pm_inputs["system_id"].drop_duplicates().to_numpy()
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=kfold_random_state)

    theta_grid_anchor = np.asarray(theta_grid, dtype=float)
    anchor_theta_map_by_dim = extract_anchor_system_means_by_dim(idata_anchor_by_dim)
    judge_ids_by_dim = {
        DIM: sorted(pm_inputs[(pm_inputs["is_llm"]) & (pm_inputs["dimension"] == DIM)]["rater_id"].dropna().unique())
        for DIM in dims
    }

    records = []
    rhat_records = []
    artifact_arrays = {} 
    artifact_manifest = [] 

    for fold_id, (train_idx, hold_idx) in tqdm(enumerate(kf.split(systems)), total=n_splits, desc="CV Folds"):
        train_systems = set(systems[train_idx])
        hold_systems = list(systems[hold_idx])

        train_df = pm_inputs[pm_inputs["system_id"].isin(train_systems)].copy()
        hold_df = pm_inputs[pm_inputs["system_id"].isin(hold_systems)].copy()
        hold_df["y_ord"] = hold_df["score"].astype(int) - 1

        for DIM in tqdm(dims, desc=f"Fold {fold_id + 1}/{n_splits}"):
            fit_result = fit_fold_models_per_dim(
                train_df=train_df,
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

            human_anchor = fit_result["human"]
            h_item_map = human_anchor["h_item_difficulty_map"]
            human_rho_map = human_anchor["human_rho_map"]
            human_cuts = human_anchor["human_cuts_mean"].squeeze()
            fold_theta_human_map = human_anchor["fold_theta_human_map"]

            overlap_systems = sorted(
                set(train_systems)
                & set(fold_theta_human_map.keys())
                & set(anchor_theta_map_by_dim[DIM].keys())
            )
            if len(overlap_systems) < 2:
                raise ValueError(f"Not enough overlap systems to link fold {fold_id}, dim {DIM}")

            theta_fold_overlap = np.array([fold_theta_human_map[s] for s in overlap_systems], dtype=float)
            theta_anchor_overlap = np.array([anchor_theta_map_by_dim[DIM][s] for s in overlap_systems], dtype=float)
            link_a, link_b, link_rmse = fit_robust_affine_link(
                theta_fold_overlap=theta_fold_overlap,
                theta_anchor_overlap=theta_anchor_overlap,
            )

            # --- persist fold-level parameters and draw subsamples
            fold_prefix = f"loocv__fold{fold_id:03d}__dim_{_safe_key_part(DIM)}"
            human_cuts_key = theta_grid_key = None
            h_item_map_json = human_rho_map_json = fold_theta_map_json = None
            anchor_theta_map_json = categories_json = None
            if save_artifacts:
                human_cuts_key = _store_array_once(
                    artifact_arrays,
                    f"{fold_prefix}__human_cutpoints",
                    np.asarray(human_cuts, dtype=np.float32),
                )
                theta_grid_key = _store_array_once(
                    artifact_arrays,
                    f"{fold_prefix}__theta_grid",
                    np.asarray(theta_grid, dtype=np.float32),
                )
                draws_sub = human_anchor.get("anchor_draw_subsample")
                if draws_sub:
                    for draw_name, draw_arr in draws_sub.items():
                        _store_array_once(
                            artifact_arrays,
                            f"{fold_prefix}__anchor_draws_{draw_name}",
                            np.asarray(draw_arr, dtype=np.float32),
                        )
                h_item_map_json = json.dumps({str(k): float(v) for k, v in h_item_map.items()}, sort_keys=True)
                human_rho_map_json = json.dumps({str(k): float(v) for k, v in human_rho_map.items()}, sort_keys=True)
                fold_theta_map_json = json.dumps(
                    {str(k): float(v) for k, v in fold_theta_human_map.items()}, sort_keys=True
                )
                anchor_theta_map_json = json.dumps(
                    {str(k): float(v) for k, v in anchor_theta_map_by_dim[DIM].items()}, sort_keys=True
                )
                categories = human_anchor.get("anchor_categories") or {}
                categories_json = json.dumps(categories, sort_keys=True)
            # -----------------------------------------------------------------

            for judge_id, judge_fit in fit_result["judges"].items():
                if judge_fit is None:
                    continue

                judge_map = judge_fit["judge_map"]
                llm_cuts = judge_fit["llm_cutpoints_mean"]

                # --- persist judge-level cutpoints and draws ------
                judge_prefix = f"{fold_prefix}__judge_{_safe_key_part(judge_id)}"
                llm_cuts_key = None
                if save_artifacts:
                    llm_cuts_key = _store_array_once(
                        artifact_arrays,
                        f"{judge_prefix}__llm_cutpoints",
                        np.asarray(llm_cuts, dtype=np.float32),
                    )
                    if judge_fit.get("llm_cuts_draws") is not None:
                        _store_array_once(
                            artifact_arrays,
                            f"{judge_prefix}__llm_cuts_draws",
                            np.asarray(judge_fit["llm_cuts_draws"], dtype=np.float32),
                        )
                # -------------------------------------------------------------

                # rhat statistics (+ divergences and min ESS)
                rhat_df = (
                    az.rhat(judge_fit["idata_llm"])
                    .to_array()
                    .to_series()
                    .dropna()
                    .sort_values(ascending=False)
                    .reset_index(name="r_hat")
                )
                n_div, min_ess = _sampler_diagnostics(judge_fit["idata_llm"])
                rhat_df["n_divergences"] = n_div
                rhat_df["min_ess"] = min_ess
                rhat_df["fold"] = fold_id
                rhat_df["dimension"] = DIM
                rhat_df["judge_model"] = judge_id
                rhat_records.append(rhat_df)

                for sid in hold_systems:
                    llm_sys = hold_df[
                        (hold_df["system_id"] == sid) &
                        (hold_df["is_llm"]) &
                        (hold_df["dimension"] == DIM) &
                        (hold_df["rater_id"] == judge_id)
                    ].copy()
                    if "score_list" in llm_sys.columns:
                        llm_sys = llm_sys[["item_id", "rater_id", "y_ord", "score_list"]]
                    else:
                        llm_sys = llm_sys[["item_id", "rater_id", "y_ord"]]

                    hum_sys = hold_df[
                        (hold_df["system_id"] == sid) &
                        (~hold_df["is_llm"]) &
                        (hold_df["dimension"] == DIM)
                    ][["item_id", "rater_id", "y_ord"]].copy()

                    if len(llm_sys) == 0 or len(hum_sys) == 0:
                        continue

                    llm_sys["score"] = llm_sys["y_ord"].astype(int) + 1
                    hum_sys["score"] = hum_sys["y_ord"].astype(int) + 1

                    post_llm_fold = get_system_posterior(
                        group_df=llm_sys,
                        judge_params=judge_map,
                        cutpoints=llm_cuts,
                        theta_grid=theta_grid,
                        item_difficulty_map=h_item_map,
                    )

                    post_h_fold = get_system_posterior_human(
                        group_df_human=hum_sys,
                        human_rater_params=human_rho_map,
                        human_cutpoints=human_cuts,
                        theta_grid=theta_grid,
                        item_difficulty_map=h_item_map,
                    )

                    post_llm = transform_posterior_to_anchor_grid(
                        post_fold=post_llm_fold,
                        theta_grid_fold=theta_grid,
                        a=link_a,
                        b=link_b,
                        theta_grid_anchor=theta_grid_anchor,
                    )
                    post_h = transform_posterior_to_anchor_grid(
                        post_fold=post_h_fold,
                        theta_grid_fold=theta_grid,
                        a=link_a,
                        b=link_b,
                        theta_grid_anchor=theta_grid_anchor,
                    )

                    theta_llm_mean = float((post_llm * theta_grid_anchor).sum())
                    theta_llm_sd = float(np.sqrt((post_llm * (theta_grid_anchor - theta_llm_mean) ** 2).sum()))
                    theta_h_mean = float((post_h * theta_grid_anchor).sum())
                    theta_h_sd = float(np.sqrt((post_h * (theta_grid_anchor - theta_h_mean) ** 2).sum()))
                    raw_llm_mean_score = float(llm_sys["score"].mean())
                    raw_human_mean_score = float(hum_sys["score"].mean())

                    dist_metrics = evaluate_system_item_jsd_emd(
                        llm_sys_df=llm_sys,
                        hum_sys_df=hum_sys,
                        theta_grid=theta_grid,
                        post_theta=post_llm_fold,
                        item_map=h_item_map,
                        human_cuts=human_cuts,
                        human_rho_map=human_rho_map,
                        K=5,
                    )

                    r = abs(theta_h_mean - theta_llm_mean)
                    r_norm = r / (theta_llm_sd + 1e-6)

                    # --- persist per-system posteriors + manifest ---
                    post_theta_key = post_human_key = None
                    post_theta_fold_key = post_human_fold_key = None
                    if save_artifacts:
                        rec_prefix = f"{judge_prefix}__system_{_safe_key_part(sid)}"
                        post_theta_key = _store_array_once(
                            artifact_arrays, f"{rec_prefix}__post_theta_anchor", post_llm.astype(np.float32)
                        )
                        post_human_key = _store_array_once(
                            artifact_arrays, f"{rec_prefix}__post_human_anchor", post_h.astype(np.float32)
                        )
                        post_theta_fold_key = _store_array_once(
                            artifact_arrays, f"{rec_prefix}__post_theta_fold", post_llm_fold.astype(np.float32)
                        )
                        post_human_fold_key = _store_array_once(
                            artifact_arrays, f"{rec_prefix}__post_human_fold", post_h_fold.astype(np.float32)
                        )
                        artifact_manifest.append({
                            "fold": fold_id,
                            "dimension": DIM,
                            "judge_model": judge_id,
                            "system_id": str(sid),
                            "post_theta_key": post_theta_key,
                            "post_human_key": post_human_key,
                            "post_theta_fold_key": post_theta_fold_key,
                            "post_human_fold_key": post_human_fold_key,
                            "theta_grid_key": theta_grid_key,
                            "human_cutpoints_key": human_cuts_key,
                            "llm_cutpoints_key": llm_cuts_key,
                            "anchor_draws_theta_key": f"{fold_prefix}__anchor_draws_theta",
                            "anchor_draws_beta_key": f"{fold_prefix}__anchor_draws_beta",
                            "anchor_draws_rho_key": f"{fold_prefix}__anchor_draws_rho",
                            "anchor_draws_cutpoints_key": f"{fold_prefix}__anchor_draws_cutpoints",
                            "llm_cuts_draws_key": f"{judge_prefix}__llm_cuts_draws",
                            "link_a": link_a,
                            "link_b": link_b,
                            "link_rmse": link_rmse,
                            "h_item_difficulty_map_json": h_item_map_json,
                            "human_rho_map_json": human_rho_map_json,
                            "fold_theta_human_map_json": fold_theta_map_json,
                            "anchor_theta_map_json": anchor_theta_map_json,
                            "anchor_categories_json": categories_json,
                        })
                    # ---------------------------------------------------------

                    records.append({
                        "fold": fold_id,
                        "dimension": DIM,
                        "system_id": sid,
                        "theta_llm": theta_llm_mean,
                        "theta_llm_sd": theta_llm_sd,
                        "theta_human": theta_h_mean,
                        "theta_human_sd": theta_h_sd,
                        "resid": r,
                        "resid_norm": r_norm,
                        "raw_llm_mean_score": raw_llm_mean_score,
                        "raw_human_mean_score": raw_human_mean_score,   
                        "post_theta": post_llm,
                        "post_human": post_h,
                        "theta_grid": theta_grid_anchor,
                        "judge_model": judge_id,
                        "link_a": link_a,
                        "link_b": link_b,
                        "link_rmse": link_rmse,
                        **dist_metrics,
                    })

    df_rec = pd.DataFrame(records)
    rhat_records = pd.concat(rhat_records, ignore_index=True)
    artifact_manifest_df = pd.DataFrame(artifact_manifest)

    q_by_dim = defaultdict(dict)
    qn_by_dim = defaultdict(dict)
    if not df_rec.empty:
        for DIM, judge in product(dims, df_rec["judge_model"].unique()):
            sub = df_rec[(df_rec["dimension"] == DIM) & (df_rec["judge_model"] == judge)]
            if len(sub) == 0:
                continue
            r = sub["resid"].values
            rn = sub["resid_norm"].values
            q_by_dim[DIM][judge] = conformal_q(r, alpha=alpha)
            qn_by_dim[DIM][judge] = conformal_q(rn, alpha=alpha)

    return df_rec, q_by_dim, qn_by_dim, rhat_records, artifact_arrays, artifact_manifest_df
