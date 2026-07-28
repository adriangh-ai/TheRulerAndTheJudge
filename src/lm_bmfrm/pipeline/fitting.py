"""Stage-1 and Stage-2 model fitting orchestration."""
from __future__ import annotations
import warnings
import jax
import numpy as np
import pandas as pd
import pymc.sampling.jax as pmjax
from lm_bmfrm.irt_model import (build_mfrm_model, build_llm_ord_model, build_llm_ord_model_mc, build_llm_ord_dif_model, build_llm_ord_dif_model_meanfield)
from lm_bmfrm.irt_utils import prepare_mfrm_data, prepare_llm_judge_data
from .posterior import extract_eta_anchor_draws, extract_eta_anchor_moments, select_anchor_draw_indices


def fit_human_anchor_for_dim(
    train_df,
    DIM,
    draws=4000,
    target_accept=0.99,
    chains=4,
    anchor_draw_selection_method="medoids",
    anchor_draw_compute_n=64,
    anchor_draw_fit_n=256,
    sampler_seed=42,
    anchor_draw_compute_seed=42,
    anchor_draw_fit_seed=123,
):
    human_train = train_df[(~train_df["is_llm"]) & (train_df["dimension"] == DIM)].copy()
    if len(human_train) == 0:
        raise ValueError(f"No human data for dimension {DIM}")

    df_h_mfrm, h_meta, h_cutpoints = prepare_mfrm_data(human_train)
    h_cutpoints = np.asarray(h_cutpoints, dtype=np.float32).squeeze()

    model_h, _ = build_mfrm_model(
        df=df_h_mfrm,
        meta=h_meta,
        fixed_cutpoints_list=np.array(h_cutpoints),
    )
    with model_h:
        idata_anchor = pmjax.sample_numpyro_nuts(
            target_accept=target_accept,
            draws=draws,
            chain_method="vectorized",
            postprocessing_backend="cpu",
            progressbar=True,
            chains=chains,
            idata_kwargs={"log_likelihood": False},
            random_seed=sampler_seed,
        )
    jax.clear_caches()

    post_h = idata_anchor.posterior
    human_beta_mean = post_h["beta"].mean(dim=["chain", "draw"]).values
    human_beta_std = post_h["beta"].std(dim=["chain", "draw"]).values
    human_theta_mean = post_h["theta"].mean(dim=["chain", "draw"]).values
    human_theta_std = post_h["theta"].std(dim=["chain", "draw"]).values
    sigma_beta_human = float(post_h["sigma_beta"].mean(dim=["chain", "draw"]).values)
    human_cuts_mean = np.asarray(post_h["cutpoints"].mean(dim=["chain", "draw"]).values).squeeze()

    rho_mean = post_h["rho_human"].mean(dim=["chain", "draw"]).values
    human_rater_names = post_h.coords["human_raters"].values
    human_rho_map = {str(name): float(rho_mean[idx]) for idx, name in enumerate(human_rater_names)}

    h_item_difficulty_map = {
        str(item_id): float(human_beta_mean[idx])
        for idx, item_id in enumerate(h_meta["item_categories"])
    }
    human_theta_map = {
        str(sys_id): float(human_theta_mean[i])
        for i, sys_id in enumerate(h_meta["system_categories"])
    }
    human_theta_std_map = {
        str(sys_id): float(human_theta_std[i])
        for i, sys_id in enumerate(h_meta["system_categories"])
    }
    human_beta_map = {
        str(item_id): float(human_beta_mean[i])
        for i, item_id in enumerate(h_meta["item_categories"])
    }
    human_beta_std_map = {
        str(item_id): float(human_beta_std[i])
        for i, item_id in enumerate(h_meta["item_categories"])
    }

    anchor_draw_compute_grid = select_anchor_draw_indices(
        idata_anchor=idata_anchor,
        n_select=anchor_draw_compute_n,
        method=anchor_draw_selection_method,
        random_state=anchor_draw_compute_seed,
    )
    anchor_draw_fit_grid = select_anchor_draw_indices(
        idata_anchor=idata_anchor,
        n_select=anchor_draw_fit_n,
        method=anchor_draw_selection_method,
        random_state=anchor_draw_fit_seed,
    )

    # --- deterministic subsample of Stage-1 posterior draws ---------
    # The subsample reuses anchor_draw_compute_grid (deterministic under the
    # default linspace selection), so persisting it consumes no RNG and adds
    # nothing to the numerical pipeline. These draws make nuisance-uncertainty
    # propagation (dominance probabilities marginalized over beta/rho/cutpoint
    # uncertainty) a pure post-processing step.
    n_total_draws_h = int(post_h.sizes["chain"] * post_h.sizes["draw"])
    g = np.asarray(anchor_draw_compute_grid, dtype=int)
    try:
        anchor_draw_subsample = {
            "theta": np.asarray(post_h["theta"]).reshape(n_total_draws_h, -1)[g].astype(np.float32),
            "beta": np.asarray(post_h["beta"]).reshape(n_total_draws_h, -1)[g].astype(np.float32),
            "rho": np.asarray(post_h["rho_human"]).reshape(n_total_draws_h, -1)[g].astype(np.float32),
            "cutpoints": np.asarray(post_h["cutpoints"]).reshape(n_total_draws_h, -1)[g].astype(np.float32),
        }
    except Exception as e:
        warnings.warn(f"Could not extract Stage-1 draw subsample for dim={DIM}: {e}")
        anchor_draw_subsample = None

    anchor_categories = {
        "system_categories": [str(s) for s in h_meta["system_categories"]],
        "item_categories": [str(i) for i in h_meta["item_categories"]],
        "rater_categories": [str(r) for r in human_rater_names],
    }
    # -------------------------------------------------------------------------

    return {
        "idata_anchor": idata_anchor,
        "meta_anchor": h_meta,
        "df_h_mfrm": df_h_mfrm,
        "h_item_difficulty_map": h_item_difficulty_map,
        "human_rho_map": human_rho_map,
        "human_cuts_mean": human_cuts_mean,
        "sigma_beta_human": sigma_beta_human,
        "fold_theta_human_map": human_theta_map,
        "human_theta_map": human_theta_map,
        "human_theta_std_map": human_theta_std_map,
        "human_beta_map": human_beta_map,
        "human_beta_std_map": human_beta_std_map,
        "anchor_draw_compute_grid": anchor_draw_compute_grid,
        "anchor_draw_fit_grid": anchor_draw_fit_grid,
        "anchor_draw_subsample": anchor_draw_subsample,
        "anchor_categories": anchor_categories,
    }


def fit_single_judge_for_dim(
    train_df,
    DIM,
    judge_id,
    human_anchor,
    draws=4000,
    target_accept=0.99,
    chains=4,
    use_mc=True,
    add_alpha=False,
    dif_params=False,
    dif_variant="mc",
    sampler_seed=42,
):
    llm_train = train_df[
        (train_df["is_llm"]) &
        (train_df["dimension"] == DIM) &
        (train_df["rater_id"] == judge_id)
    ].copy()

    if len(llm_train) == 0:
        return None

    df_llm, meta_llm, q_obs, obs_weight = prepare_llm_judge_data(
        llm_train,
        n_classes=5,
        aggregate_repeats=False,
    )

    eta_anchor_mean, eta_anchor_sd = extract_eta_anchor_moments(
        idata_anchor=human_anchor["idata_anchor"],
        target_system_categories=meta_llm["system_categories"],
        target_item_categories=meta_llm["item_categories"],
    )

    eta_anchor_draws = extract_eta_anchor_draws(
        idata_anchor=human_anchor["idata_anchor"],
        target_system_categories=meta_llm["system_categories"],
        target_item_categories=meta_llm["item_categories"],
        anchor_draw_compute_grid=human_anchor["anchor_draw_compute_grid"],
    )

    if use_mc:
        model_llm, _ = build_llm_ord_model_mc(
            df=df_llm,
            meta=meta_llm,
            eta_anchor_draws=eta_anchor_draws,
            human_cutpoints_mean=human_anchor["human_cuts_mean"],
            q_obs=q_obs,
            obs_weight=obs_weight,
            add_alpha=add_alpha,
        )
    else:
        model_llm, _ = build_llm_ord_model(
            df=df_llm,
            meta=meta_llm,
            eta_anchor_mean=eta_anchor_mean,
            eta_anchor_sd=eta_anchor_sd,
            human_cutpoints_mean=human_anchor["human_cuts_mean"],
            q_obs=q_obs,
            obs_weight=obs_weight,
            add_alpha=add_alpha,
        )

    with model_llm:
        idata_llm = pmjax.sample_numpyro_nuts(
            target_accept=target_accept,
            draws=draws,
            chain_method="vectorized",
            postprocessing_backend="cpu",
            progressbar=True,
            chains=chains,
            idata_kwargs={"log_likelihood": False},
            random_seed=sampler_seed,
        )
    jax.clear_caches()

    llm_beta_diff_df = pd.DataFrame()
    if dif_params:
        # the default DIF model marginalizes over the SAME Stage-1
        # anchor draws and uses the SAME soft-label likelihood as the main
        # Stage-2 model, so it is a strict nested extension (main model +
        # judge-by-item beta_diff). The legacy mean-field plug-in variant is
        # retained for A/B comparison via dif_variant="meanfield".
        if str(dif_variant) == "mc":
            model_dif, _ = build_llm_ord_dif_model(
                df=df_llm,
                meta=meta_llm,
                eta_anchor_draws=eta_anchor_draws,
                human_cutpoints_mean=human_anchor["human_cuts_mean"],
                q_obs=q_obs,
                obs_weight=obs_weight,
            )
        elif str(dif_variant) == "meanfield":
            aligned_theta = np.array(
                [human_anchor["human_theta_map"][str(s)] for s in meta_llm["system_categories"]],
                dtype=np.float32,
            )
            aligned_theta_std = np.array(
                [human_anchor["human_theta_std_map"][str(s)] for s in meta_llm["system_categories"]],
                dtype=np.float32,
            )
            aligned_beta = np.array(
                [human_anchor["human_beta_map"][str(i)] for i in meta_llm["item_categories"]],
                dtype=np.float32,
            )
            aligned_beta_std = np.array(
                [human_anchor["human_beta_std_map"][str(i)] for i in meta_llm["item_categories"]],
                dtype=np.float32,
            )
            model_dif, _ = build_llm_ord_dif_model_meanfield(
                df=df_llm,
                meta=meta_llm,
                human_beta=aligned_beta,
                human_theta=aligned_theta,
                human_beta_std=aligned_beta_std,
                human_theta_std=aligned_theta_std,
                human_cutpoints_mean=human_anchor["human_cuts_mean"],
            )
        else:
            raise ValueError(f"Unknown dif_variant: {dif_variant!r} (expected 'mc' or 'meanfield').")
        with model_dif:
            idata_llm_dif = pmjax.sample_numpyro_nuts(
                target_accept=target_accept,
                draws=draws,
                chain_method="vectorized",
                postprocessing_backend="cpu",
                progressbar=True,
                chains=chains,
                idata_kwargs={"log_likelihood": False},
                random_seed=sampler_seed,
            )
        jax.clear_caches()

        post_l_dif = idata_llm_dif.posterior
        llm_beta_diff_mean = post_l_dif["beta_diff"].mean(dim=["chain", "draw"]).values
        llm_beta_diff_df = pd.DataFrame(
            llm_beta_diff_mean,
            index=post_l_dif.coords["llm_raters"],
            columns=post_l_dif.coords["items"],
        )

    post_l = idata_llm.posterior
    llm_cutpoints_mean = post_l["llm_cuts"].mean(dim=["chain", "draw"]).values

    llm_rater_names = post_l.coords["llm_raters"].values
    judge_map = {str(name): {"rater_idx": int(i)} for i, name in enumerate(llm_rater_names)}

    # --- deterministic subsample of Stage-2 judge cutpoint draws ----
    n_total_draws_l = int(post_l.sizes["chain"] * post_l.sizes["draw"])
    try:
        cuts_draws_all = np.asarray(post_l["llm_cuts"]).reshape(n_total_draws_l, -1)
        sel = np.unique(np.linspace(0, n_total_draws_l - 1, min(200, n_total_draws_l), dtype=int))
        llm_cuts_draws = cuts_draws_all[sel].astype(np.float32)
    except Exception as e:
        warnings.warn(f"Could not extract Stage-2 cutpoint draws for judge={judge_id}, dim={DIM}: {e}")
        llm_cuts_draws = None

    # --- alpha posterior summary if a discrimination variant ran ---
    alpha_summary = None
    for cand in ("alpha", "log_alpha", "alpha_judge", "log_alpha_judge", "alpha_m", "log_alpha_m"):
        if cand in post_l:
            try:
                arr = np.asarray(post_l[cand]).reshape(n_total_draws_l, -1)
                alpha_summary = {
                    "var_name": cand,
                    "mean": [float(x) for x in arr.mean(axis=0)],
                    "sd": [float(x) for x in arr.std(axis=0)],
                }
            except Exception:
                alpha_summary = {"var_name": cand, "mean": None, "sd": None}
            break
    # -------------------------------------------------------------------------

    return {
        "judge_id": judge_id,
        "idata_llm": idata_llm,
        "meta_llm": meta_llm,
        "df_llm": df_llm,
        "llm_cutpoints_mean": llm_cutpoints_mean,
        "judge_map": judge_map,
        "llm_beta_diff": llm_beta_diff_df,
        "llm_cuts_draws": llm_cuts_draws,
        "alpha_summary": alpha_summary,
    }


def fit_fold_models_per_dim(
    train_df,
    DIM,
    judge_ids=None,
    draws=4000,
    target_accept=0.99,
    chains=4,
    dif_params=False,
    anchor_draw_selection_method="medoids",
    anchor_draw_compute_n=64,
    anchor_draw_fit_n=256,
    use_mc=True,
    add_alpha=False,
    dif_variant="mc",
    sampler_seed=42,
    anchor_draw_compute_seed=42,
    anchor_draw_fit_seed=123,
):
    human_anchor = fit_human_anchor_for_dim(
        train_df=train_df,
        DIM=DIM,
        draws=draws,
        target_accept=target_accept,
        chains=chains,
        anchor_draw_selection_method=anchor_draw_selection_method,
        anchor_draw_compute_n=anchor_draw_compute_n,
        anchor_draw_fit_n=anchor_draw_fit_n,
        sampler_seed=sampler_seed,
        anchor_draw_compute_seed=anchor_draw_compute_seed,
        anchor_draw_fit_seed=anchor_draw_fit_seed,
    )

    if judge_ids is None:
        judge_ids = sorted(train_df[(train_df["is_llm"]) & (train_df["dimension"] == DIM)]["rater_id"].dropna().unique())

    judge_models = {}
    for judge_id in judge_ids:
        print(f"Fitting judge model for dimension {DIM}, judge {judge_id}...")
        judge_models[judge_id] = fit_single_judge_for_dim(
            train_df=train_df,
            DIM=DIM,
            judge_id=judge_id,
            human_anchor=human_anchor,
            draws=draws,
            target_accept=target_accept,
            chains=chains,
            use_mc=use_mc,
            add_alpha=add_alpha,
            dif_params=dif_params,
            dif_variant=dif_variant,
            sampler_seed=sampler_seed,
        )

    return {
        "human": human_anchor,
        "judges": judge_models,
    }
