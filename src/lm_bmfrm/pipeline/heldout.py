"""Held-out system prediction from fitted fold models."""
from __future__ import annotations
import json
import numpy as np
from lm_bmfrm.evaluation_metrics import evaluate_system_item_jsd_emd
from lm_bmfrm.estimate import get_system_posterior, get_system_posterior_human
from .cv_common import _extract_rhat_records, _posterior_mean_sd, _safe_key_part, _store_array_once
from .linking import transform_posterior_to_anchor_grid


def _evaluate_heldout_systems_from_fit(
    *,
    fit_result,
    eval_df,
    hold_systems,
    DIM,
    theta_grid,
    theta_grid_anchor,
    link_a,
    link_b,
    link_rmse,
    outer_fold,
    eval_split,
    inner_fold,
    train_system_count,
    artifact_arrays,
    artifact_manifest,
):
    """
    Estimate held-out system theta posteriors for all judges in one fitted dim model.

    The returned records contain posterior means/SDs and artifact keys. The actual
    posterior vectors and cutpoint vectors are stored in artifact_arrays so they can
    be persisted once in a compressed NPZ file.
    """
    human_anchor = fit_result["human"]
    h_item_map = human_anchor["h_item_difficulty_map"]
    human_rho_map = human_anchor["human_rho_map"]
    human_cuts = np.asarray(human_anchor["human_cuts_mean"]).squeeze()

    records = []
    rhat_records = []
    theta_grid = np.asarray(theta_grid, dtype=float)
    theta_grid_anchor = np.asarray(theta_grid_anchor, dtype=float)

    dim_fit_prefix = (
        f"outer{outer_fold:03d}__{_safe_key_part(eval_split)}__"
        f"inner{_safe_key_part('final' if inner_fold is None else inner_fold)}__"
        f"dim_{_safe_key_part(DIM)}"
    )
    human_cuts_key = _store_array_once(
        artifact_arrays,
        f"{dim_fit_prefix}__human_cutpoints",
        human_cuts.astype(np.float32),
    )
    theta_grid_fold_key = _store_array_once(
        artifact_arrays,
        f"{dim_fit_prefix}__theta_grid_fold",
        theta_grid.astype(np.float32),
    )
    theta_grid_anchor_key = _store_array_once(
        artifact_arrays,
        f"{dim_fit_prefix}__theta_grid_anchor",
        theta_grid_anchor.astype(np.float32),
    )

    #  (nested-mode parity): persist Stage-1/Stage-2 draw subsamples too.
    draws_sub = human_anchor.get("anchor_draw_subsample")
    if draws_sub:
        for draw_name, draw_arr in draws_sub.items():
            _store_array_once(
                artifact_arrays,
                f"{dim_fit_prefix}__anchor_draws_{draw_name}",
                np.asarray(draw_arr, dtype=np.float32),
            )

    h_item_map_json = json.dumps({str(k): float(v) for k, v in h_item_map.items()}, sort_keys=True)
    human_rho_map_json = json.dumps({str(k): float(v) for k, v in human_rho_map.items()}, sort_keys=True)

    for judge_id, judge_fit in fit_result["judges"].items():
        if judge_fit is None:
            continue

        judge_map = judge_fit["judge_map"]
        llm_cuts = np.asarray(judge_fit["llm_cutpoints_mean"])
        judge_fit_prefix = f"{dim_fit_prefix}__judge_{_safe_key_part(judge_id)}"
        llm_cuts_key = _store_array_once(
            artifact_arrays,
            f"{judge_fit_prefix}__llm_cutpoints",
            llm_cuts.astype(np.float32),
        )
        if judge_fit.get("llm_cuts_draws") is not None:
            _store_array_once(
                artifact_arrays,
                f"{judge_fit_prefix}__llm_cuts_draws",
                np.asarray(judge_fit["llm_cuts_draws"], dtype=np.float32),
            )

        rhat_records.append(_extract_rhat_records(
            idata_llm=judge_fit["idata_llm"],
            outer_fold=outer_fold,
            eval_split=eval_split,
            dimension=DIM,
            judge_model=judge_id,
            inner_fold=inner_fold,
        ))

        for sid in hold_systems:
            llm_sys = eval_df[
                (eval_df["system_id"].astype(str) == str(sid)) &
                (eval_df["is_llm"]) &
                (eval_df["dimension"] == DIM) &
                (eval_df["rater_id"] == judge_id)
            ].copy()
            if "score_list" in llm_sys.columns:
                llm_sys = llm_sys[["item_id", "rater_id", "y_ord", "score_list"]]
            else:
                llm_sys = llm_sys[["item_id", "rater_id", "y_ord"]]

            hum_sys = eval_df[
                (eval_df["system_id"].astype(str) == str(sid)) &
                (~eval_df["is_llm"]) &
                (eval_df["dimension"] == DIM)
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

            theta_llm_mean, theta_llm_sd = _posterior_mean_sd(post_llm, theta_grid_anchor)
            theta_h_mean, theta_h_sd = _posterior_mean_sd(post_h, theta_grid_anchor)
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

            rec_prefix = f"{judge_fit_prefix}__system_{_safe_key_part(sid)}"
            post_theta_key = _store_array_once(
                artifact_arrays,
                f"{rec_prefix}__post_theta_anchor",
                post_llm.astype(np.float32),
            )
            post_human_key = _store_array_once(
                artifact_arrays,
                f"{rec_prefix}__post_human_anchor",
                post_h.astype(np.float32),
            )
            post_theta_fold_key = _store_array_once(
                artifact_arrays,
                f"{rec_prefix}__post_theta_fold",
                post_llm_fold.astype(np.float32),
            )
            post_human_fold_key = _store_array_once(
                artifact_arrays,
                f"{rec_prefix}__post_human_fold",
                post_h_fold.astype(np.float32),
            )

            artifact_manifest.append({
                "outer_fold": outer_fold,
                "inner_fold": inner_fold,
                "eval_split": eval_split,
                "dimension": DIM,
                "judge_model": judge_id,
                "system_id": str(sid),
                "post_theta_key": post_theta_key,
                "post_human_key": post_human_key,
                "post_theta_fold_key": post_theta_fold_key,
                "post_human_fold_key": post_human_fold_key,
                "theta_grid_fold_key": theta_grid_fold_key,
                "theta_grid_anchor_key": theta_grid_anchor_key,
                "human_cutpoints_key": human_cuts_key,
                "llm_cutpoints_key": llm_cuts_key,
                "h_item_difficulty_map_json": h_item_map_json,
                "human_rho_map_json": human_rho_map_json,
            })

            records.append({
                "outer_fold": outer_fold,
                "inner_fold": inner_fold,
                "eval_split": eval_split,
                "fold": inner_fold if inner_fold is not None else -1,
                "dimension": DIM,
                "system_id": str(sid),
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
                "post_theta_key": post_theta_key,
                "post_human_key": post_human_key,
                "post_theta_fold_key": post_theta_fold_key,
                "post_human_fold_key": post_human_fold_key,
                "theta_grid_fold_key": theta_grid_fold_key,
                "theta_grid_anchor_key": theta_grid_anchor_key,
                "human_cutpoints_key": human_cuts_key,
                "llm_cutpoints_key": llm_cuts_key,
                "judge_model": judge_id,
                "link_a": link_a,
                "link_b": link_b,
                "link_rmse": link_rmse,
                "train_system_count": int(train_system_count),
                "heldout_system_count": int(len(hold_systems)),
                **dist_metrics,
            })

    return records, rhat_records
