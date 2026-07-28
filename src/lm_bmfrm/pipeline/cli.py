"""Command-line orchestration for the complete evaluation pipeline."""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime
import json
import os
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from rich.console import Console
from rich.table import Table

from lm_bmfrm.data_preprocess import load_all_human_judgments, load_hanna, load_hitz_basse
from .config import ExperimentConfig, load_experiment_config
from .data_quality import (
    build_judge_coverage_table,
    load_judge_outputs,
    resolve_dimensions,
    resolve_judge_files,
    validate_judgment_dataframe,
)


def _preparse_config(argv=None) -> tuple[str | None, ExperimentConfig]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--config", type=str, default=None)
    known, _ = parser.parse_known_args(argv)
    return known.config, load_experiment_config(known.config)


def _build_parser(config: ExperimentConfig) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the Stage 1--3 psychometric evaluation pipeline.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="JSON configuration merged over the packaged paper defaults.",
    )
    parser.add_argument(
        "--benchmark",
        choices=["summeval", "summeval_synth", "hitz_basse", "hanna", "hitz_basse_vas"],
        default=config.data.benchmark,
        required=config.data.benchmark is None,
        help="Dataset to benchmark.",
    )
    parser.add_argument(
        "--judge_folder",
        type=str,
        default=config.data.judge_folder,
        help="Root folder containing one benchmark subfolder of judge parquet files.",
    )
    parser.add_argument(
        "--judge_file",
        action="append",
        dest="judge_files",
        default=None,
        help="Explicit judge parquet path. Repeat to define a deterministic file order.",
    )
    parser.add_argument(
        "--dimensions",
        nargs="+",
        default=None,
        help="Explicit ordered dimension subset. Otherwise first-seen judge-output order is retained.",
    )
    parser.add_argument(
        "--anchor_draw_selection",
        choices=["medoids", "linspace"],
        default=config.model.anchor_draw_selection,
        help="Selection method for propagated Stage-1 posterior draws.",
    )
    parser.add_argument(
        "--cv_mode",
        choices=["nested", "loocv"],
        default=config.evaluation.cv_mode,
        help="Stage-3 validation mode.",
    )
    parser.add_argument("--draws", type=int, default=config.sampling.draws)
    parser.add_argument("--chains_audit", type=int, default=config.sampling.chains_audit)
    parser.add_argument("--chains_cv", type=int, default=config.sampling.chains_cv)
    parser.add_argument("--target_accept", type=float, default=config.sampling.target_accept)
    parser.add_argument("--sampler_seed", type=int, default=config.sampling.sampler_seed)
    parser.add_argument(
        "--anchor_draw_compute_n", type=int, default=config.sampling.anchor_draw_compute_n
    )
    parser.add_argument("--anchor_draw_fit_n", type=int, default=config.sampling.anchor_draw_fit_n)
    parser.add_argument(
        "--anchor_draw_compute_seed", type=int, default=config.sampling.anchor_draw_compute_seed
    )
    parser.add_argument(
        "--anchor_draw_fit_seed", type=int, default=config.sampling.anchor_draw_fit_seed
    )
    parser.add_argument("--theta_min", type=float, default=config.evaluation.theta_grid.min)
    parser.add_argument("--theta_max", type=float, default=config.evaluation.theta_grid.max)
    parser.add_argument("--theta_num", type=int, default=config.evaluation.theta_grid.num)
    parser.add_argument("--tau_h", type=float, default=config.evaluation.tau_h)
    parser.add_argument("--tau_l", type=float, default=config.evaluation.tau_l)
    parser.add_argument("--conformal_alpha", type=float, default=config.evaluation.conformal_alpha)
    parser.add_argument("--kfold_seed", type=int, default=config.evaluation.kfold_seed)

    parser.add_argument("--outer_test_size", type=float, default=config.nested_cv.outer_test_size)
    parser.add_argument("--outer_splits", type=int, default=config.nested_cv.outer_splits)
    parser.add_argument("--outer_seed", type=int, default=config.nested_cv.outer_seed)
    parser.add_argument(
        "--outer_split_strategy",
        choices=["random", "stratified_theta", "constrained_blocked"],
        default=config.nested_cv.outer_split_strategy,
    )
    parser.add_argument(
        "--outer_stratify_dimension",
        type=str,
        default=config.nested_cv.outer_stratify_dimension,
    )
    parser.add_argument(
        "--outer_constraint_tau_h", type=float, default=config.nested_cv.outer_constraint_tau_h
    )
    parser.add_argument(
        "--outer_min_supported_pairs",
        type=int,
        default=config.nested_cv.outer_min_supported_pairs,
    )
    parser.add_argument(
        "--outer_max_supported_pairs",
        type=int,
        default=config.nested_cv.outer_max_supported_pairs,
    )
    parser.add_argument("--outer_min_sr_h", type=float, default=config.nested_cv.outer_min_sr_h)
    parser.add_argument("--outer_max_sr_h", type=float, default=config.nested_cv.outer_max_sr_h)
    parser.add_argument(
        "--outer_min_spread_fraction",
        type=float,
        default=config.nested_cv.outer_min_spread_fraction,
    )
    parser.add_argument(
        "--outer_max_candidate_splits",
        type=int,
        default=config.nested_cv.outer_max_candidate_splits,
    )
    parser.add_argument(
        "--no_save_artifacts",
        dest="save_artifacts",
        action="store_false",
        default=config.outputs.save_artifacts,
        help="Disable LOOCV/nested posterior artifact persistence.",
    )
    parser.add_argument(
        "--no_save_idata",
        dest="save_idata",
        action="store_false",
        default=config.outputs.save_idata,
        help="Disable full-data Stage-1/Stage-2 InferenceData persistence.",
    )
    parser.add_argument(
        "--results_root",
        type=str,
        default=config.outputs.results_root,
        help="Directory under which timestamped run folders are created.",
    )
    parser.add_argument(
        "--audit_add_alpha",
        action="store_true",
        default=config.model.audit_add_alpha,
        help="Fit the optional judge-discrimination audit variant.",
    )
    parser.add_argument(
        "--dif_variant",
        choices=["mc", "meanfield"],
        default=config.model.dif_variant,
        help="DIF model used in the full-data audit.",
    )
    return parser


def _validate_cli_args(args: argparse.Namespace) -> None:
    """Validate command-line overrides before loading data or probabilistic code."""
    if args.theta_num <= 1 or args.theta_min >= args.theta_max:
        raise ValueError("theta grid requires theta_num > 1 and theta_min < theta_max.")

    positive_int_fields = {
        "draws": args.draws,
        "chains_audit": args.chains_audit,
        "chains_cv": args.chains_cv,
        "anchor_draw_compute_n": args.anchor_draw_compute_n,
        "anchor_draw_fit_n": args.anchor_draw_fit_n,
        "outer_splits": args.outer_splits,
        "outer_max_candidate_splits": args.outer_max_candidate_splits,
    }
    invalid_positive = [name for name, value in positive_int_fields.items() if value <= 0]
    if invalid_positive:
        raise ValueError(f"These arguments must be positive: {', '.join(invalid_positive)}.")

    open_probabilities = {
        "target_accept": args.target_accept,
        "conformal_alpha": args.conformal_alpha,
        "outer_test_size": args.outer_test_size,
    }
    for name, value in open_probabilities.items():
        if not 0.0 < value < 1.0:
            raise ValueError(f"{name} must be in (0, 1); got {value}.")

    closed_probabilities = {
        "tau_h": args.tau_h,
        "tau_l": args.tau_l,
        "outer_constraint_tau_h": args.outer_constraint_tau_h,
    }
    optional_closed_probabilities = {
        "outer_min_sr_h": args.outer_min_sr_h,
        "outer_max_sr_h": args.outer_max_sr_h,
    }
    for name, value in {**closed_probabilities, **optional_closed_probabilities}.items():
        if value is not None and not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be in [0, 1]; got {value}.")

    if (
        args.outer_min_sr_h is not None
        and args.outer_max_sr_h is not None
        and args.outer_min_sr_h > args.outer_max_sr_h
    ):
        raise ValueError("outer_min_sr_h cannot exceed outer_max_sr_h.")
    if args.outer_min_spread_fraction < 0.0:
        raise ValueError("outer_min_spread_fraction must be nonnegative.")
    if (
        args.outer_min_supported_pairs >= 0
        and args.outer_max_supported_pairs >= 0
        and args.outer_min_supported_pairs > args.outer_max_supported_pairs
    ):
        raise ValueError("outer_min_supported_pairs cannot exceed outer_max_supported_pairs.")


def _load_human_data(benchmark: str) -> pd.DataFrame:
    if benchmark in {"summeval", "summeval_synth"}:
        frame = load_all_human_judgments(use_summeval_hf=True)
    elif benchmark == "hanna":
        frame = load_hanna()
    elif benchmark == "hitz_basse":
        frame = load_hitz_basse()
    elif benchmark == "hitz_basse_vas":
        frame = load_hitz_basse(split="eu")
    else:  # guarded by argparse/config validation
        raise ValueError(f"Unsupported benchmark: {benchmark}")
    frame = frame.copy()
    frame["is_llm"] = False
    validate_judgment_dataframe(frame, source=f"human benchmark {benchmark}")
    return frame


def _resolved_config_dict(args, base_config: ExperimentConfig, judge_files: list[str], dims: list[str]):
    config = base_config.to_dict()
    config["data"].update(
        {
            "benchmark": args.benchmark,
            "judge_folder": args.judge_folder,
            "judge_files": judge_files,
            "dimensions": dims,
        }
    )
    config["model"].update(
        {
            "anchor_draw_selection": args.anchor_draw_selection,
            "audit_add_alpha": args.audit_add_alpha,
            "dif_variant": args.dif_variant,
        }
    )
    config["sampling"].update(
        {
            "draws": args.draws,
            "chains_audit": args.chains_audit,
            "chains_cv": args.chains_cv,
            "target_accept": args.target_accept,
            "sampler_seed": args.sampler_seed,
            "anchor_draw_compute_n": args.anchor_draw_compute_n,
            "anchor_draw_fit_n": args.anchor_draw_fit_n,
            "anchor_draw_compute_seed": args.anchor_draw_compute_seed,
            "anchor_draw_fit_seed": args.anchor_draw_fit_seed,
        }
    )
    config["evaluation"].update(
        {
            "cv_mode": args.cv_mode,
            "theta_grid": {"min": args.theta_min, "max": args.theta_max, "num": args.theta_num},
            "tau_h": args.tau_h,
            "tau_l": args.tau_l,
            "conformal_alpha": args.conformal_alpha,
            "kfold_seed": args.kfold_seed,
        }
    )
    config["nested_cv"].update(
        {
            "outer_test_size": args.outer_test_size,
            "outer_splits": args.outer_splits,
            "outer_seed": args.outer_seed,
            "outer_split_strategy": args.outer_split_strategy,
            "outer_stratify_dimension": args.outer_stratify_dimension,
            "outer_constraint_tau_h": args.outer_constraint_tau_h,
            "outer_min_supported_pairs": args.outer_min_supported_pairs,
            "outer_max_supported_pairs": args.outer_max_supported_pairs,
            "outer_min_sr_h": args.outer_min_sr_h,
            "outer_max_sr_h": args.outer_max_sr_h,
            "outer_min_spread_fraction": args.outer_min_spread_fraction,
            "outer_max_candidate_splits": args.outer_max_candidate_splits,
        }
    )
    config["outputs"].update(
        {
            "results_root": args.results_root,
            "save_artifacts": args.save_artifacts,
            "save_idata": args.save_idata,
        }
    )
    return config


def main(argv=None):
    config_path, base_config = _preparse_config(argv)
    args = _build_parser(base_config).parse_args(argv)

    _validate_cli_args(args)

    explicit_files = args.judge_files
    if explicit_files is None and base_config.data.judge_files is not None:
        explicit_files = list(base_config.data.judge_files)
    configured_dimensions = args.dimensions
    if configured_dimensions is None and base_config.data.dimensions is not None:
        configured_dimensions = list(base_config.data.dimensions)

    judge_files = resolve_judge_files(
        judge_folder=args.judge_folder,
        benchmark=args.benchmark,
        explicit_files=explicit_files,
    )
    df_human = _load_human_data(args.benchmark)
    judge_df, exclusions_df, judge_manifest_df = load_judge_outputs(judge_files)
    dims = resolve_dimensions(judge_df, args.benchmark, configured_dimensions)

    judge_df = judge_df.reset_index(drop=True)
    judge_df["dataset"] = args.benchmark
    judge_df["score"] = judge_df["score"].astype(int)
    judge_df["is_llm"] = True

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{args.benchmark}_MC_per_judge_{args.anchor_draw_selection}_{timestamp}"
    results_dir = os.path.join(args.results_root, run_name)
    os.makedirs(results_dir, exist_ok=True)

    exclusions_df.to_csv(os.path.join(results_dir, "judge_exclusions.csv"), index=False)
    judge_manifest_df.to_parquet(
        os.path.join(results_dir, "judge_input_manifest.parquet"), index=False
    )
    coverage_df = build_judge_coverage_table(judge_df, df_human, dims)
    coverage_df.to_csv(os.path.join(results_dir, "judge_coverage.csv"), index=False)

    theta_grid = np.linspace(args.theta_min, args.theta_max, args.theta_num)
    resolved_config = _resolved_config_dict(args, base_config, judge_files, dims)
    Path(results_dir, "resolved_config.json").write_text(
        json.dumps(resolved_config, indent=2, sort_keys=True), encoding="utf-8"
    )

    # Heavy probabilistic imports occur only after argument parsing and input validation.
    from .artifacts import (
        _jsonify_q_flat,
        _jsonify_qdict,
        parquet_safe_predictions,
        run_judge_benchmark,
        save_prediction_artifacts,
    )
    from .audit import run_benchmark_audit
    from .cv_common import _extract_full_human_pairwise_by_dim, _extract_full_human_theta_by_dim
    from .leaderboard import build_psychometric_leaderboard, summarize_outer_leaderboard
    from .nested_cv import run_nested_system_cv_conformal
    from .provenance import write_run_config

    write_run_config(
        results_dir,
        args,
        extra={
            "config_path": config_path,
            "resolved_config": resolved_config,
            "n_judge_files": len(judge_files),
        },
    )

    audit_results = run_benchmark_audit(
        judge_df=judge_df,
        human_df=df_human,
        dims=dims,
        theta_grid=theta_grid,
        target_accept=args.target_accept,
        draws=args.draws,
        chains=args.chains_audit,
        anchor_draw_selection_method=args.anchor_draw_selection,
        anchor_draw_compute_n=args.anchor_draw_compute_n,
        anchor_draw_fit_n=args.anchor_draw_fit_n,
        results_dir=results_dir,
        save_idata=args.save_idata,
        audit_add_alpha=args.audit_add_alpha,
        dif_variant=args.dif_variant,
        use_mc=base_config.model.use_mc,
        add_alpha_main=base_config.model.add_alpha_main,
        sampler_seed=args.sampler_seed,
        anchor_draw_compute_seed=args.anchor_draw_compute_seed,
        anchor_draw_fit_seed=args.anchor_draw_fit_seed,
    )

    nested_results = None
    df_oof = pd.DataFrame()
    rhat_records = pd.DataFrame()
    df_lb = pd.DataFrame()
    df_lb_inner = pd.DataFrame()
    df_lb_test = pd.DataFrame()
    df_lb_summary = pd.DataFrame()
    loocv_artifact_arrays = {}
    loocv_artifact_manifest = pd.DataFrame()

    if args.cv_mode == "nested":
        pm_inputs = pd.concat([df_human, judge_df], ignore_index=True)
        min_supported_pairs = (
            None if args.outer_min_supported_pairs < 0 else args.outer_min_supported_pairs
        )
        max_supported_pairs = (
            None if args.outer_max_supported_pairs < 0 else args.outer_max_supported_pairs
        )
        nested_results = run_nested_system_cv_conformal(
            pm_inputs=pm_inputs,
            dims=dims,
            theta_grid=theta_grid,
            outer_test_size=args.outer_test_size,
            n_outer_splits=args.outer_splits,
            outer_random_state=args.outer_seed,
            outer_split_strategy=args.outer_split_strategy,
            full_human_theta_by_dim=_extract_full_human_theta_by_dim(
                audit_results.get("human_anchor")
            ),
            full_human_pairwise_by_dim=_extract_full_human_pairwise_by_dim(
                audit_results.get("human_anchor")
            ),
            outer_stratify_dimension=args.outer_stratify_dimension,
            outer_constrained_tau_h=args.outer_constraint_tau_h,
            outer_min_supported_pairs=min_supported_pairs,
            outer_max_supported_pairs=max_supported_pairs,
            outer_min_sr_h=args.outer_min_sr_h,
            outer_max_sr_h=args.outer_max_sr_h,
            outer_min_spread_fraction=args.outer_min_spread_fraction,
            outer_max_candidate_splits=args.outer_max_candidate_splits,
            alpha=args.conformal_alpha,
            draws_cv=args.draws,
            chains_cv=args.chains_cv,
            target_accept=args.target_accept,
            anchor_draw_selection_method=args.anchor_draw_selection,
            anchor_draw_compute_n=args.anchor_draw_compute_n,
            anchor_draw_fit_n=args.anchor_draw_fit_n,
            use_mc=base_config.model.use_mc,
            add_alpha_main=base_config.model.add_alpha_main,
            sampler_seed=args.sampler_seed,
            anchor_draw_compute_seed=args.anchor_draw_compute_seed,
            anchor_draw_fit_seed=args.anchor_draw_fit_seed,
        )
        df_oof = nested_results["predictions"]
        rhat_records = nested_results["rhat_records"]
        q_raw = nested_results["q_raw_by_outer"]
        q_norm = nested_results["q_norm_by_outer"]
        df_lb_inner = build_psychometric_leaderboard(
            df_oof,
            q_norm=q_norm,
            q_raw=q_raw,
            theta_grid=theta_grid,
            split_name="inner_loocv",
            tau_h=args.tau_h,
            tau_l=args.tau_l,
        )
        df_lb_test = build_psychometric_leaderboard(
            df_oof,
            q_norm=q_norm,
            q_raw=q_raw,
            theta_grid=theta_grid,
            split_name="outer_test",
            tau_h=args.tau_h,
            tau_l=args.tau_l,
        )
        df_lb = pd.concat([df_lb_inner, df_lb_test], ignore_index=True)
        df_lb_summary = summarize_outer_leaderboard(df_lb)
    else:
        (
            df_oof,
            q_raw,
            q_norm,
            rhat_records,
            loocv_artifact_arrays,
            loocv_artifact_manifest,
        ) = run_judge_benchmark(
            judge_df,
            df_human,
            dims,
            idata_anchor_by_dim=audit_results["idata_anchor"],
            theta_grid=theta_grid,
            n_total_draws=args.draws,
            n_splits=len(df_human.system_id.unique()),
            target_accept=args.target_accept,
            anchor_draw_selection_method=args.anchor_draw_selection,
            anchor_draw_compute_n=args.anchor_draw_compute_n,
            anchor_draw_fit_n=args.anchor_draw_fit_n,
            save_artifacts=args.save_artifacts,
            alpha=args.conformal_alpha,
            chains_cv=args.chains_cv,
            use_mc=base_config.model.use_mc,
            add_alpha_main=base_config.model.add_alpha_main,
            sampler_seed=args.sampler_seed,
            anchor_draw_compute_seed=args.anchor_draw_compute_seed,
            anchor_draw_fit_seed=args.anchor_draw_fit_seed,
            kfold_random_state=args.kfold_seed,
        )
        df_oof["outer_fold"] = 0
        df_oof["inner_fold"] = df_oof.get("fold", np.nan)
        df_oof["eval_split"] = "loocv"
        q_raw_nested = defaultdict(lambda: defaultdict(dict))
        q_norm_nested = defaultdict(lambda: defaultdict(dict))
        q_raw_nested[0] = q_raw
        q_norm_nested[0] = q_norm
        df_lb = build_psychometric_leaderboard(
            df_oof,
            q_norm=q_norm_nested,
            q_raw=q_raw_nested,
            theta_grid=theta_grid,
            split_name="loocv",
            tau_h=args.tau_h,
            tau_l=args.tau_l,
        )

    table = Table(title="LLM-as-a-Judge Psychometric Leaderboard")
    if df_lb.empty:
        table.add_column("Message")
        table.add_row("No leaderboard rows were produced.")
    else:
        for column in df_lb.columns:
            table.add_column(column)
        for _, row in df_lb.iterrows():
            table.add_row(*[f"{value:.3f}" if isinstance(value, float) else str(value) for value in row])
    Console().print(table)

    pd.concat(audit_results["human_audit"]).reset_index().to_parquet(
        os.path.join(results_dir, "human_audit.parquet"), index=False
    )
    pd.concat(audit_results["human_item_audit"]).reset_index().to_parquet(
        os.path.join(results_dir, "human_item_audit.parquet"), index=False
    )
    pd.DataFrame(audit_results["benchmark_cards"]).to_parquet(
        os.path.join(results_dir, "benchmark_cards.parquet"), index=False
    )
    pd.DataFrame(audit_results["benchmark_curves"]).to_parquet(
        os.path.join(results_dir, "benchmark_curves.parquet"), index=False
    )
    if audit_results["llm_audit"]:
        pd.concat(audit_results["llm_audit"]).reset_index().to_parquet(
            os.path.join(results_dir, "llm_audit.parquet"), index=False
        )

    for dimension, judge_dict in audit_results["dif_audit"].items():
        for judge_id, dif_audit in judge_dict.items():
            if isinstance(dif_audit, pd.DataFrame) and not dif_audit.empty:
                safe_judge = str(judge_id).replace("/", "__")
                dif_audit.to_parquet(
                    os.path.join(results_dir, f"dif_audit_{dimension}_{safe_judge}.parquet")
                )
    if audit_results.get("alpha_audit"):
        pd.DataFrame(audit_results["alpha_audit"]).to_parquet(
            os.path.join(results_dir, "alpha_audit.parquet"), index=False
        )

    df_lb.to_parquet(os.path.join(results_dir, "leaderboard_results.parquet"), index=False)
    if not df_lb_inner.empty:
        df_lb_inner.to_parquet(
            os.path.join(results_dir, "inner_loocv_leaderboard.parquet"), index=False
        )
    if not df_lb_test.empty:
        df_lb_test.to_parquet(
            os.path.join(results_dir, "outer_test_leaderboard.parquet"), index=False
        )
    if not df_lb_summary.empty:
        df_lb_summary.to_parquet(
            os.path.join(results_dir, "leaderboard_outer_summary.parquet"), index=False
        )

    parquet_safe_predictions(df_oof).to_parquet(
        os.path.join(results_dir, "predictions.parquet"), index=False
    )
    if args.cv_mode == "nested" and nested_results is not None:
        nested_results["split_manifest"].to_parquet(
            os.path.join(results_dir, "outer_split_manifest.parquet"), index=False
        )
        if args.save_artifacts:
            save_prediction_artifacts(nested_results, results_dir, prefix="nested")
        q_raw_json = _jsonify_qdict(nested_results["q_raw_by_outer"])
        q_norm_json = _jsonify_qdict(nested_results["q_norm_by_outer"])
        for filename, payload in {
            "q_raw_by_outer.json": q_raw_json,
            "q_norm_by_outer.json": q_norm_json,
            "q_raw_inner_loocv_calibration_by_outer.json": q_raw_json,
            "q_norm_inner_loocv_calibration_by_outer.json": q_norm_json,
        }.items():
            Path(results_dir, filename).write_text(
                json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
            )

    if args.cv_mode == "loocv":
        if loocv_artifact_arrays and args.save_artifacts:
            try:
                np.savez_compressed(
                    os.path.join(results_dir, "loocv_posterior_artifacts.npz"),
                    **loocv_artifact_arrays,
                )
            except Exception as exc:
                warnings.warn(f"Could not save LOOCV posterior artifacts: {exc}")
        if isinstance(loocv_artifact_manifest, pd.DataFrame) and not loocv_artifact_manifest.empty:
            loocv_artifact_manifest.to_parquet(
                os.path.join(results_dir, "loocv_posterior_artifacts_manifest.parquet"),
                index=False,
            )
        Path(results_dir, "q_raw_loocv.json").write_text(
            json.dumps(_jsonify_q_flat(q_raw), indent=2, sort_keys=True), encoding="utf-8"
        )
        Path(results_dir, "q_norm_loocv.json").write_text(
            json.dumps(_jsonify_q_flat(q_norm), indent=2, sort_keys=True), encoding="utf-8"
        )

    rhat_records.to_parquet(os.path.join(results_dir, "rhat_records.parquet"), index=False)
    if audit_results["llm_item_audit"]:
        pd.concat(audit_results["llm_item_audit"]).reset_index().to_parquet(
            os.path.join(results_dir, "llm_item_audit.parquet"), index=False
        )
