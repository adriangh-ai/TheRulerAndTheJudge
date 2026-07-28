"""Full-data benchmark and judge audit."""
from __future__ import annotations
from collections import defaultdict
import json, os, warnings
import numpy as np
import pandas as pd
from rich.console import Console
from tqdm import tqdm
from lm_bmfrm.fit_analysis import (calculate_rasch_fit_statistics_human_items, calculate_rasch_fit_statistics_step2, print_benchmark_resolution_card, print_rasch_fit_table, summarize_benchmark_resolution)
from .cv_common import _safe_key_part
from .fitting import fit_fold_models_per_dim, fit_single_judge_for_dim
from .provenance import _sampler_diagnostics


def run_benchmark_audit(
    judge_df,
    human_df,
    dims,
    theta_grid,
    target_accept=0.99,
    draws=4000,
    chains=4,
    anchor_draw_selection_method="medoids",
    anchor_draw_compute_n=64,
    anchor_draw_fit_n=256,
    results_dir=None,
    save_idata=True,
    audit_add_alpha=False,
    dif_variant="mc",
    use_mc=True,
    add_alpha_main=False,
    sampler_seed=42,
    anchor_draw_compute_seed=42,
    anchor_draw_fit_seed=123,
):
    console = Console()
    pm_inputs = pd.concat([human_df, judge_df], ignore_index=True)
    pm_inputs["y_ord"] = pm_inputs["score"].astype(int) - 1

    human_audit, human_item_audit, llm_audit = [], [], []
    all_llm_item_audit = []
    benchmark_cards = []
    benchmark_curves = pd.DataFrame()
    idata_dim_map = {}
    anchor_system_categories = {}
    dif_audit_map = defaultdict(dict)
    human_anchor_map = {}
    alpha_audit_rows = []

    judge_ids_by_dim = {
        DIM: sorted(judge_df[judge_df["dimension"] == DIM]["rater_id"].dropna().unique())
        for DIM in dims
    }

    for DIM in tqdm(dims, desc="Auditing benchmark and judges"):
        fit_result = fit_fold_models_per_dim(
            train_df=pm_inputs,
            DIM=DIM,
            judge_ids=judge_ids_by_dim[DIM],
            draws=draws,
            target_accept=target_accept,
            chains=chains,
            dif_params=False,
            anchor_draw_selection_method=anchor_draw_selection_method,
            anchor_draw_compute_n=anchor_draw_compute_n,
            anchor_draw_fit_n=anchor_draw_fit_n,
            use_mc=use_mc,
            add_alpha=add_alpha_main,
            dif_variant=dif_variant,
            sampler_seed=sampler_seed,
            anchor_draw_compute_seed=anchor_draw_compute_seed,
            anchor_draw_fit_seed=anchor_draw_fit_seed,
        )

        human_anchor = fit_result["human"]
        human_anchor_map[DIM] = human_anchor

        # --- persist full-data Stage-1 idata + parameter JSON -------
        if results_dir is not None:
            if save_idata:
                try:
                    human_anchor["idata_anchor"].to_netcdf(
                        os.path.join(results_dir, f"stage1_idata_{_safe_key_part(DIM)}.nc")
                    )
                except Exception as e:
                    warnings.warn(f"Could not save Stage-1 idata for dim={DIM}: {e}")
            try:
                stage1_params = {
                    "dimension": str(DIM),
                    "human_cuts_mean": [float(x) for x in np.atleast_1d(human_anchor["human_cuts_mean"])],
                    "h_item_difficulty_map": human_anchor["h_item_difficulty_map"],
                    "human_rho_map": human_anchor["human_rho_map"],
                    "human_theta_map": human_anchor["human_theta_map"],
                    "human_theta_std_map": human_anchor["human_theta_std_map"],
                    "human_beta_std_map": human_anchor["human_beta_std_map"],
                    "sigma_beta_human": float(human_anchor["sigma_beta_human"]),
                    "anchor_categories": human_anchor.get("anchor_categories"),
                }
                with open(os.path.join(results_dir, f"stage1_params_{_safe_key_part(DIM)}.json"), "w", encoding="utf-8") as f:
                    json.dump(stage1_params, f, indent=2, sort_keys=True)
            except Exception as e:
                warnings.warn(f"Could not save Stage-1 params for dim={DIM}: {e}")
        # ---------------------------------------------------------------------

        item_fit, rater_fit = calculate_rasch_fit_statistics_human_items(
            df=human_anchor["df_h_mfrm"],
            idata=human_anchor["idata_anchor"],
            meta=human_anchor["meta_anchor"],
        )

        resolution_summary, curve_df, _ = summarize_benchmark_resolution(
            df=human_anchor["df_h_mfrm"],
            idata=human_anchor["idata_anchor"],
            meta=human_anchor["meta_anchor"],
            theta_grid=theta_grid,
            central_mass=0.80,
            z_value=1.96,
            average_per_system=True,
        )

        # print_rasch_fit_table(item_fit, console, type="Human Items", dim=DIM)
        print_rasch_fit_table(rater_fit, console, type="Human Raters", dim=DIM)
        print_benchmark_resolution_card(resolution_summary, console, dim=DIM)

        item_fit["dimension"] = DIM
        rater_fit["dimension"] = DIM
        resolution_summary["dimension"] = DIM
        curve_df["dimension"] = DIM

        human_item_audit.append(item_fit)
        human_audit.append(rater_fit)
        benchmark_cards.append(resolution_summary)
        benchmark_curves = pd.concat([benchmark_curves, curve_df], ignore_index=True)

        dim_llm_audit = []
        dim_llm_item_audit = []
        for judge_id, judge_fit in fit_result["judges"].items():
            if judge_fit is None:
                continue

            llm_mnsq = calculate_rasch_fit_statistics_step2(
                df=judge_fit["df_llm"],
                idata=judge_fit["idata_llm"],
                meta=judge_fit["meta_llm"],
                idata_anchor=human_anchor["idata_anchor"],
                anchor_draw_grid=human_anchor["anchor_draw_fit_grid"],
            )
            llm_fit = llm_mnsq["rater_fit"]
            llm_fit["dimension"] = DIM
            llm_fit["judge_model"] = judge_id
            llm_audit.append(llm_fit)

            llm_item_fit = llm_mnsq["item_fit"]
            llm_item_fit["dimension"] = DIM
            llm_item_fit["judge_model"] = judge_id
            dim_llm_item_audit.append(llm_item_fit)

            dim_llm_audit.append(llm_fit)
            dif_audit_map[DIM][judge_id] = judge_fit["llm_beta_diff"]

            # --- persist full-data Stage-2 idata per judge ----------
            if results_dir is not None and save_idata:
                try:
                    judge_fit["idata_llm"].to_netcdf(
                        os.path.join(
                            results_dir,
                            f"stage2_idata_{_safe_key_part(DIM)}__{_safe_key_part(judge_id)}.nc",
                        )
                    )
                except Exception as e:
                    warnings.warn(f"Could not save Stage-2 idata for dim={DIM}, judge={judge_id}: {e}")
            # -----------------------------------------------------------------

        # --- optional judge-discrimination (alpha) audit variant ---
        if audit_add_alpha:
            for judge_id in judge_ids_by_dim[DIM]:
                print(f"[alpha variant] Fitting add_alpha judge model for dimension {DIM}, judge {judge_id}...")
                judge_fit_alpha = fit_single_judge_for_dim(
                    train_df=pm_inputs,
                    DIM=DIM,
                    judge_id=judge_id,
                    human_anchor=human_anchor,
                    draws=draws,
                    target_accept=target_accept,
                    chains=chains,
                    use_mc=use_mc,
                    add_alpha=True,
                    dif_params=False,
                    sampler_seed=sampler_seed,
                )
                if judge_fit_alpha is None:
                    continue
                row = {
                    "dimension": DIM,
                    "judge_model": judge_id,
                }
                if judge_fit_alpha.get("alpha_summary") is not None:
                    row.update({
                        "alpha_var_name": judge_fit_alpha["alpha_summary"].get("var_name"),
                        "alpha_mean_json": json.dumps(judge_fit_alpha["alpha_summary"].get("mean")),
                        "alpha_sd_json": json.dumps(judge_fit_alpha["alpha_summary"].get("sd")),
                    })
                else:
                    row.update({"alpha_var_name": None, "alpha_mean_json": None, "alpha_sd_json": None})
                row["llm_cutpoints_mean_json"] = json.dumps(
                    [float(x) for x in np.asarray(judge_fit_alpha["llm_cutpoints_mean"]).ravel()]
                )
                n_div, min_ess = _sampler_diagnostics(judge_fit_alpha["idata_llm"])
                row["n_divergences"] = n_div
                row["min_ess"] = min_ess
                alpha_audit_rows.append(row)
                if results_dir is not None and save_idata:
                    try:
                        judge_fit_alpha["idata_llm"].to_netcdf(
                            os.path.join(
                                results_dir,
                                f"stage2_idata_alpha_{_safe_key_part(DIM)}__{_safe_key_part(judge_id)}.nc",
                            )
                        )
                    except Exception as e:
                        warnings.warn(f"Could not save alpha-variant idata for dim={DIM}, judge={judge_id}: {e}")
        # ---------------------------------------------------------------------

        print_rasch_fit_table(pd.concat(dim_llm_audit), console, type=f"LLM Judge", dim=DIM)
        idata_dim_map[DIM] = human_anchor["idata_anchor"]
        anchor_system_categories[DIM] = human_anchor["meta_anchor"]["system_categories"]

        all_llm_item_audit.extend(dim_llm_item_audit)

    return {
        "human_audit": human_audit,
        "human_item_audit": human_item_audit,
        "llm_audit": llm_audit,
        "benchmark_cards": benchmark_cards,
        "benchmark_curves": benchmark_curves,
        "dif_audit": dif_audit_map,
        "idata_anchor": idata_dim_map,
        "anchor_system_categories": anchor_system_categories,
        "human_anchor": human_anchor_map,
        "llm_item_audit": all_llm_item_audit,
        "alpha_audit": alpha_audit_rows,
    }
