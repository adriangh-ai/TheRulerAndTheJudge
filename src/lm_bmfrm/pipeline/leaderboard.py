"""Decision metrics and psychometric leaderboard construction."""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from lm_bmfrm.evaluation_metrics import (compute_false_separation_rate, compute_nsmf_from_generative_probabilities, compute_pairwise_generative_probabilities_vectorized, compute_pairwise_probabilities)


def _lookup_nested_q(qdict, outer_fold, dim, judge):
    outer_fold = int(outer_fold)
    if outer_fold in qdict:
        return qdict[outer_fold][dim][judge]
    return qdict[dim][judge]


def _as_pairwise_dict(pairwise_obj):
    """Normalize pairwise probability outputs to {(a,b): p}."""
    if isinstance(pairwise_obj, dict):
        out = {}
        for k, v in pairwise_obj.items():
            if isinstance(k, tuple) and len(k) == 2:
                out[(str(k[0]), str(k[1]))] = float(v)
            elif isinstance(k, str):
                # Best-effort parser for keys such as "a||b", "a|b", or "a,b".
                for sep in ["||", "|", ",", "__"]:
                    if sep in k:
                        a, b = k.split(sep, 1)
                        out[(str(a), str(b))] = float(v)
                        break
        return out

    if isinstance(pairwise_obj, pd.Series):
        return _as_pairwise_dict(pairwise_obj.to_dict())

    if isinstance(pairwise_obj, pd.DataFrame):
        cols = list(pairwise_obj.columns)
        a_col = next((c for c in cols if str(c).lower() in {"a", "system_a", "system_i", "left"}), None)
        b_col = next((c for c in cols if str(c).lower() in {"b", "system_b", "system_j", "right"}), None)
        p_col = next((c for c in cols if str(c).lower() in {"p", "prob", "probability", "pairwise_p"}), None)
        if a_col is not None and b_col is not None and p_col is not None:
            return {(str(r[a_col]), str(r[b_col])): float(r[p_col]) for _, r in pairwise_obj.iterrows()}

    raise TypeError(f"Unsupported pairwise object type: {type(pairwise_obj)}")


def _safe_div(num, den):
    den = float(den)
    return float(num) / den if den > 0 else np.nan


def compute_support_confusion_metrics(human_pairwise_p, judge_pairwise_p, tau_h=0.8, tau_l=0.8):
    """
    Compute support-confusion diagnostics for risk-sensitive pairwise decisions.

    H_ab=1 iff |2p_H(a>b)-1| >= tau_h.
    J_ab=1 iff |2p_J(a>b)-1| >= tau_l.
    """
    hp = _as_pairwise_dict(human_pairwise_p)
    jp = _as_pairwise_dict(judge_pairwise_p)
    keys = sorted(set(hp).intersection(jp))
    n = len(keys)
    if n == 0:
        return {
            "N pairs": 0,
            "Human SR": np.nan,
            "Judge SR": np.nan,
            "SR ratio": np.nan,
            "Support TP": 0,
            "Support FP": 0,
            "Support FN": 0,
            "Support TN": 0,
            "Support precision": np.nan,
            "Support recall": np.nan,
            "Support FSR": np.nan,
            "Unsupported Separation Rate": np.nan,
            "Supported Direction Accuracy": np.nan,
            "PairAcc tau": np.nan,
            "Human pairwise baseline": np.nan,
            "Human mean abs margin": np.nan,
        }

    tp = fp = fn = tn = 0
    supported_direction_correct = 0
    supported_both = 0
    tie_aware_correct = 0
    human_supported_rank_correct = 0
    human_supported_rank_total = 0

    actionable_correct = 0
    judge_supported_total = 0
    human_supported_total = 0

    h_margin_sq = []
    h_abs_margin = []

    for k in keys:
        mh = 2.0 * float(hp[k]) - 1.0
        mj = 2.0 * float(jp[k]) - 1.0
        h_margin_sq.append(mh * mh)
        h_abs_margin.append(abs(mh))

        h_sup = abs(mh) >= tau_h
        j_sup = abs(mj) >= tau_l
        if h_sup and j_sup:
            tp += 1
            supported_both += 1
            if np.sign(mh) == np.sign(mj):
                supported_direction_correct += 1
        elif (not h_sup) and j_sup:
            fp += 1
        elif h_sup and (not j_sup):
            fn += 1
        else:
            tn += 1

        # Tie-aware decision code: +1, -1, or 0 unsupported/tie.
        if h_sup:
            dh = 1 if mh > 0 else -1
        else:
            dh = 0
        if j_sup:
            dj = 1 if mj > 0 else -1
        else:
            dj = 0
        if dh == dj:
            tie_aware_correct += 1

        # Directional support recall and precision
        sign_h = np.sign(mh)
        sign_j = np.sign(mj)

        if h_sup:
            human_supported_rank_total += 1
            if sign_h == sign_j:
                human_supported_rank_correct += 1

        if j_sup:
            judge_supported_total += 1

        if h_sup:
            human_supported_total += 1

        if h_sup and j_sup and sign_h == sign_j:
            actionable_correct += 1

    human_sr = _safe_div(tp + fn, n)
    judge_sr = _safe_div(tp + fp, n)
    return {
        "N pairs": int(n),
        "Human SR": human_sr,
        "Judge SR": judge_sr,
        "SR ratio": _safe_div(judge_sr, human_sr) if np.isfinite(human_sr) and human_sr > 0 else np.nan,
        "Support TP": int(tp),
        "Support FP": int(fp),
        "Support FN": int(fn),
        "Support TN": int(tn),
        "Support precision": _safe_div(tp, tp + fp),
        "Support recall": _safe_div(tp, tp + fn),
        "Support FSR": _safe_div(fp, fp + tn),
        "Unsupported Separation Rate": _safe_div(fn, tp + fn),
        "Supported Direction Accuracy": _safe_div(supported_direction_correct, supported_both),
        "PairAcc tau": _safe_div(tie_aware_correct, n),
        "Human pairwise baseline": float(np.mean(h_margin_sq)),
        "Human mean abs margin": float(np.mean(h_abs_margin)),
        "Human-supported rank accuracy": _safe_div(
            human_supported_rank_correct,
            human_supported_rank_total,
        ),
        "Directional support precision": _safe_div(
            actionable_correct,
            judge_supported_total,
        ),
        "Directional support recall": _safe_div(
            actionable_correct,
            human_supported_total,
        ),
    }


def compute_pairwise_rank_accuracy(theta_human, theta_judge, eps=1e-12):
    """Pairwise sign agreement between human and judge posterior means."""
    human = {str(k): float(v) for k, v in theta_human.items()}
    judge = {str(k): float(v) for k, v in theta_judge.items()}
    systems = sorted(set(human).intersection(judge))
    if len(systems) < 2:
        return np.nan
    correct = 0
    total = 0
    for i, a in enumerate(systems):
        for b in systems[i + 1:]:
            dh = human[a] - human[b]
            dj = judge[a] - judge[b]
            sh = 0 if abs(dh) < eps else int(np.sign(dh))
            sj = 0 if abs(dj) < eps else int(np.sign(dj))
            correct += int(sh == sj)
            total += 1
    return _safe_div(correct, total)


def build_psychometric_leaderboard(
    df_pred,
    q_norm,
    q_raw,
    theta_grid,
    split_name=None,
    tau_h=0.8,
    tau_l=0.8,
):
    """Compute the same pairwise and interval metrics for LOOCV or test predictions."""
    if df_pred is None or df_pred.empty:
        return pd.DataFrame()

    work = df_pred.copy()
    if split_name is not None:
        work = work[work["eval_split"] == split_name].copy()
    if work.empty:
        return pd.DataFrame()

    leaderboard_rows = []
    group_cols = ["outer_fold", "judge_model", "dimension"] if "outer_fold" in work.columns else ["judge_model", "dimension"]
    for group_key, sub_df in work.groupby(group_cols):
        if len(group_cols) == 3:
            outer_fold, judge, dim_target = group_key
        else:
            judge, dim_target = group_key
            outer_fold = 0

        sub_df = sub_df.sort_values("system_id").drop_duplicates(subset=["system_id"], keep="first")
        if len(sub_df) < 2:
            continue

        human_posteriors_by_system = {
            str(row["system_id"]): np.asarray(row["post_human"], dtype=np.float64)
            for _, row in sub_df.iterrows()
        }
        judge_posteriors_by_system = {
            str(row["system_id"]): np.asarray(row["post_theta"], dtype=np.float64)
            for _, row in sub_df.iterrows()
        }

        human_pairwise_p = compute_pairwise_probabilities(
            posteriors_by_system=human_posteriors_by_system,
            theta_grid=theta_grid,
        )
        judge_pairwise_p = compute_pairwise_probabilities(
            posteriors_by_system=judge_posteriors_by_system,
            theta_grid=theta_grid,
        )
        fsr = compute_false_separation_rate(
            human_pairwise_p,
            judge_pairwise_p,
            tau_h=tau_h,
            tau_l=tau_l,
        )
        support_metrics = compute_support_confusion_metrics(
            human_pairwise_p,
            judge_pairwise_p,
            tau_h=tau_h,
            tau_l=tau_l,
        )

        human_pairwise_g = compute_pairwise_generative_probabilities_vectorized(
            posteriors_by_system=human_posteriors_by_system,
            theta_grid=theta_grid,
        )
        judge_pairwise_g = compute_pairwise_generative_probabilities_vectorized(
            posteriors_by_system=judge_posteriors_by_system,
            theta_grid=theta_grid,
        )
        gen_metrics = compute_nsmf_from_generative_probabilities(
            human_pairwise_g=human_pairwise_g,
            judge_pairwise_g=judge_pairwise_g,
        )

        validity_rho = spearmanr(sub_df["theta_llm"], sub_df["theta_human"]).correlation
        raw_score_spearman = (
            spearmanr(sub_df["raw_llm_mean_score"], sub_df["raw_human_mean_score"]).correlation
            if "raw_human_mean_score" in sub_df.columns else np.nan
        )
        pairacc_theta = compute_pairwise_rank_accuracy(
            theta_human={str(r["system_id"]): float(r["theta_human"]) for _, r in sub_df.iterrows()},
            theta_judge={str(r["system_id"]): float(r["theta_llm"]) for _, r in sub_df.iterrows()},
        )
        qn = _lookup_nested_q(q_norm, outer_fold, dim_target, judge)
        qr = _lookup_nested_q(q_raw, outer_fold, dim_target, judge)

        avg_width = float((2.0 * qn * (sub_df["theta_llm_sd"] + 1e-6)).mean())
        avg_width_raw = float(2.0 * qr)
        fidelity_emd = float(sub_df["emd_pred"].mean()) if "emd_pred" in sub_df.columns else np.nan

        low = sub_df["theta_llm"] - qn * (sub_df["theta_llm_sd"] + 1e-6)
        high = sub_df["theta_llm"] + qn * (sub_df["theta_llm_sd"] + 1e-6)
        cov = float(np.mean((sub_df["theta_human"] >= low) & (sub_df["theta_human"] <= high)))

        low_raw = sub_df["theta_llm"] - qr
        high_raw = sub_df["theta_llm"] + qr
        cov_raw = float(np.mean((sub_df["theta_human"] >= low_raw) & (sub_df["theta_human"] <= high_raw)))

        leaderboard_rows.append({
            "Eval split": split_name if split_name is not None else "all",
            "Outer fold": int(outer_fold),
            "Model": judge,
            "Dimension": dim_target,
            "N systems": int(len(sub_df)),
            "NSMF": gen_metrics["nsmf"],
            "NSMF MSE": gen_metrics["signed_margin_mse"],
            "NSMF Baseline": gen_metrics["signed_margin_baseline"],
            "NSMF Uncertainty MSE": gen_metrics["uncertainty_mse"],
            "NSMF Direction MSE": gen_metrics["direction_mse"],
            "NSMF Uncertainty Loss": gen_metrics["uncertainty_loss"],
            "NSMF Direction Loss": gen_metrics["direction_loss"],
            "False Separation Rate": fsr,
            "Support FSR": support_metrics["Support FSR"],
            "Human SR": support_metrics["Human SR"],
            "Judge SR": support_metrics["Judge SR"],
            "SR ratio": support_metrics["SR ratio"],
            "Support precision": support_metrics["Support precision"],
            "Support recall": support_metrics["Support recall"],
            "Unsupported Separation Rate": support_metrics["Unsupported Separation Rate"],
            "Supported Direction Accuracy": support_metrics["Supported Direction Accuracy"],
            "PairAcc tau": support_metrics["PairAcc tau"],
            "PairAcc theta": pairacc_theta,
            "Support TP": support_metrics["Support TP"],
            "Support FP": support_metrics["Support FP"],
            "Support FN": support_metrics["Support FN"],
            "Support TN": support_metrics["Support TN"],
            "N pairs": support_metrics["N pairs"],
            "Human pairwise baseline": support_metrics["Human pairwise baseline"],
            "Human mean abs margin": support_metrics["Human mean abs margin"],
            "Spearman": validity_rho,
            "Raw-score Spearman": raw_score_spearman,
            "Error Interval norm": avg_width,
            "Coverage norm": cov,
            "Error Interval raw": avg_width_raw,
            "Coverage raw": cov_raw,
            "EMD": fidelity_emd,
        })

    return pd.DataFrame(leaderboard_rows)


def summarize_outer_leaderboard(df_lb):
    """Mean/std summary across repeated outer splits, if n_outer_splits > 1."""
    if df_lb is None or df_lb.empty:
        return pd.DataFrame()
    metric_cols = [
        c for c in df_lb.columns
        if c not in {"Eval split", "Outer fold", "Model", "Dimension"}
        and pd.api.types.is_numeric_dtype(df_lb[c])
    ]
    grouped = df_lb.groupby(["Eval split", "Model", "Dimension"], dropna=False)[metric_cols]
    mean_df = grouped.mean().add_suffix(" mean")
    std_df = grouped.std(ddof=0).add_suffix(" std")
    return pd.concat([mean_df, std_df], axis=1).reset_index()
