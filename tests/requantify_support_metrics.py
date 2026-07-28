"""
Requantify Stage-3 support diagnostics under the corrected (mid-p) tie handling,
directly from stored LOOCV posterior artifacts. NO MODEL REFITTING.

Usage:
    python requantify_support_metrics.py --results_dir results/<run_dir> \
        [--tau_h 0.8] [--tau_l 0.8] [--out_prefix tie_correction]

Reads:
    <results_dir>/loocv_posterior_artifacts.npz
    <results_dir>/loocv_posterior_artifacts_manifest.parquet

Writes:
    <out_prefix>_support_metrics_both.csv   per (dimension, judge): all support
                                            metrics under strict-discard vs mid-p,
                                            plus deltas
    <out_prefix>_pair_flips.csv             every pair whose support/direction
                                            decision changes, with both margins
                                            and tie mass e
    <out_prefix>_corrected_pairwise.parquet corrected per-pair p values (human and
                                            judge) for reuse in tau sweeps,
                                            jackknife, and practical-impact counts

Also prints a decision-stability summary: which (dimension, judge) cells change
any headline rate at the second decimal, and whether the per-dimension judge
ordering by FSR / SR changes.

Self-contained: does not import the project package.
"""

import argparse
import os
from itertools import combinations

import numpy as np
import pandas as pd


# ----------------------------------------------------------------------------
# Dominance probabilities: strict-discard (as published) and mid-p (corrected)
# ----------------------------------------------------------------------------

def _dominance_components(pa, pb):
    """Return (p_gt, p_eq) for normalized mass vectors on a shared sorted grid."""
    pa = np.asarray(pa, dtype=np.float64)
    pb = np.asarray(pb, dtype=np.float64)
    pa = pa / pa.sum()
    pb = pb / pb.sum()
    cdf_b = np.cumsum(pb)
    below = np.concatenate([[0.0], cdf_b[:-1]])
    p_gt = float(np.dot(pa, below))
    p_eq = float(np.dot(pa, pb))
    return p_gt, p_eq


def dominance_strict(pa, pb):
    p_gt, _ = _dominance_components(pa, pb)
    return p_gt


def dominance_midp(pa, pb):
    p_gt, p_eq = _dominance_components(pa, pb)
    return p_gt + 0.5 * p_eq


# ----------------------------------------------------------------------------
# Support confusion metrics (mirrors the pipeline's compute_support_confusion_metrics)
# ----------------------------------------------------------------------------

def support_confusion(hp, jp, tau_h, tau_l):
    keys = sorted(set(hp).intersection(jp))
    n = len(keys)
    tp = fp = fn = tn = 0
    sda_num = sda_den = 0
    tie_correct = 0
    for k in keys:
        mh = 2.0 * hp[k] - 1.0
        mj = 2.0 * jp[k] - 1.0
        h_sup = abs(mh) >= tau_h
        j_sup = abs(mj) >= tau_l
        if h_sup and j_sup:
            tp += 1
            sda_den += 1
            if np.sign(mh) == np.sign(mj):
                sda_num += 1
        elif (not h_sup) and j_sup:
            fp += 1
        elif h_sup and (not j_sup):
            fn += 1
        else:
            tn += 1
        dh = (1 if mh > 0 else -1) if h_sup else 0
        dj = (1 if mj > 0 else -1) if j_sup else 0
        tie_correct += int(dh == dj)

    def sdiv(a, b):
        return float(a) / b if b > 0 else np.nan

    return {
        "N pairs": n,
        "Human SR": sdiv(tp + fn, n),
        "Judge SR": sdiv(tp + fp, n),
        "Support FSR": sdiv(fp, fp + tn),
        "Support precision": sdiv(tp, tp + fp),
        "Support recall": sdiv(tp, tp + fn),
        "Unsupported Separation Rate": sdiv(fn, tp + fn),
        "Supported Direction Accuracy": sdiv(sda_num, sda_den),
        "PairAcc tau": sdiv(tie_correct, n),
        "Support TP": tp, "Support FP": fp, "Support FN": fn, "Support TN": tn,
    }


# ----------------------------------------------------------------------------
# Artifact loading
# ----------------------------------------------------------------------------

def load_group_posteriors(npz, manifest, dimension, judge):
    sub = manifest[(manifest["dimension"] == dimension) & (manifest["judge_model"] == judge)]
    sub = sub.sort_values("system_id").drop_duplicates(subset=["system_id"], keep="first")
    if len(sub) < 2:
        return None
    grid_key = sub.iloc[0]["theta_grid_key"]
    theta_grid = np.asarray(npz[grid_key], dtype=np.float64)
    human, judge_p = {}, {}
    for _, row in sub.iterrows():
        sid = str(row["system_id"])
        human[sid] = np.asarray(npz[row["post_human_key"]], dtype=np.float64)
        judge_p[sid] = np.asarray(npz[row["post_theta_key"]], dtype=np.float64)
    return theta_grid, human, judge_p


def pairwise_maps(posteriors, fn):
    ids = sorted(posteriors)
    return {(a, b): fn(posteriors[a], posteriors[b]) for a, b in combinations(ids, 2)}


def tie_mass_map(posteriors):
    ids = sorted(posteriors)
    return {(a, b): _dominance_components(posteriors[a], posteriors[b])[1]
            for a, b in combinations(ids, 2)}


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", required=True)
    ap.add_argument("--tau_h", type=float, default=0.8)
    ap.add_argument("--tau_l", type=float, default=0.8)
    ap.add_argument("--out_prefix", default="tie_correction")
    args = ap.parse_args()

    npz_path = os.path.join(args.results_dir, "loocv_posterior_artifacts.npz")
    man_path = os.path.join(args.results_dir, "loocv_posterior_artifacts_manifest.parquet")
    if not (os.path.exists(npz_path) and os.path.exists(man_path)):
        raise SystemExit(
            f"Artifacts not found in {args.results_dir}. This script requires the "
            "patched-pipeline LOOCV run (loocv_posterior_artifacts.npz + manifest)."
        )

    npz = np.load(npz_path)
    manifest = pd.read_parquet(man_path)

    rows, flips, corrected_pairs = [], [], []

    groups = manifest[["dimension", "judge_model"]].drop_duplicates()
    for _, g in groups.iterrows():
        dim, judge = g["dimension"], g["judge_model"]
        loaded = load_group_posteriors(npz, manifest, dim, judge)
        if loaded is None:
            continue
        theta_grid, hum_post, jud_post = loaded

        hp_strict = pairwise_maps(hum_post, dominance_strict)
        jp_strict = pairwise_maps(jud_post, dominance_strict)
        hp_midp = pairwise_maps(hum_post, dominance_midp)
        jp_midp = pairwise_maps(jud_post, dominance_midp)
        e_h = tie_mass_map(hum_post)
        e_j = tie_mass_map(jud_post)

        m_strict = support_confusion(hp_strict, jp_strict, args.tau_h, args.tau_l)
        m_midp = support_confusion(hp_midp, jp_midp, args.tau_h, args.tau_l)

        row = {"dimension": dim, "judge_model": judge}
        for k in m_strict:
            row[f"{k} (strict)"] = m_strict[k]
            row[f"{k} (midp)"] = m_midp[k]
            if isinstance(m_strict[k], float) and isinstance(m_midp[k], float):
                row[f"{k} (delta)"] = (m_midp[k] - m_strict[k]
                                       if np.isfinite(m_midp[k]) and np.isfinite(m_strict[k])
                                       else np.nan)
        row["mean tie mass (human)"] = float(np.mean(list(e_h.values())))
        row["mean tie mass (judge)"] = float(np.mean(list(e_j.values())))
        row["max tie mass (human)"] = float(np.max(list(e_h.values())))
        row["max tie mass (judge)"] = float(np.max(list(e_j.values())))
        rows.append(row)

        for pair in hp_strict:
            def decision(p, tau):
                m = 2.0 * p - 1.0
                return (1 if m > 0 else -1) if abs(m) >= tau else 0
            dh_s = decision(hp_strict[pair], args.tau_h)
            dh_m = decision(hp_midp[pair], args.tau_h)
            dj_s = decision(jp_strict[pair], args.tau_l)
            dj_m = decision(jp_midp[pair], args.tau_l)
            if dh_s != dh_m or dj_s != dj_m:
                flips.append({
                    "dimension": dim, "judge_model": judge,
                    "system_a": pair[0], "system_b": pair[1],
                    "human decision strict->midp": f"{dh_s}->{dh_m}",
                    "judge decision strict->midp": f"{dj_s}->{dj_m}",
                    "human p strict": hp_strict[pair], "human p midp": hp_midp[pair],
                    "judge p strict": jp_strict[pair], "judge p midp": jp_midp[pair],
                    "tie mass human": e_h[pair], "tie mass judge": e_j[pair],
                })
            corrected_pairs.append({
                "dimension": dim, "judge_model": judge,
                "system_a": pair[0], "system_b": pair[1],
                "human_p_midp": hp_midp[pair], "judge_p_midp": jp_midp[pair],
                "human_p_strict": hp_strict[pair], "judge_p_strict": jp_strict[pair],
                "tie_mass_human": e_h[pair], "tie_mass_judge": e_j[pair],
            })

    df = pd.DataFrame(rows)
    df_flips = pd.DataFrame(flips)
    df_pairs = pd.DataFrame(corrected_pairs)

    df.to_csv(f"{args.out_prefix}_support_metrics_both.csv", index=False)
    df_flips.to_csv(f"{args.out_prefix}_pair_flips.csv", index=False)
    df_pairs.to_parquet(f"{args.out_prefix}_corrected_pairwise.parquet", index=False)

    # ---- Decision-stability summary ----------------------------------------
    print(f"\n{len(df)} (dimension, judge) cells requantified.")
    print(f"Pair-level decision flips: {len(df_flips)} "
          f"(of {len(df_pairs)} total pairs).")

    headline = ["Support FSR", "Human SR", "Judge SR", "PairAcc tau",
                "Supported Direction Accuracy"]
    for k in headline:
        d = df[f"{k} (delta)"].abs()
        n_big = int((d >= 0.01).sum())
        print(f"{k}: max |delta| = {d.max():.4f}; cells with |delta| >= 0.01: {n_big}")

    # Judge ordering stability per dimension by corrected vs strict FSR
    print("\nPer-dimension judge ordering by Support FSR:")
    for dim, sub in df.groupby("dimension"):
        order_s = list(sub.sort_values("Support FSR (strict)")["judge_model"])
        order_m = list(sub.sort_values("Support FSR (midp)")["judge_model"])
        stable = order_s == order_m
        print(f"  {dim}: {'STABLE' if stable else 'CHANGED'}")
        if not stable:
            print(f"    strict: {order_s}")
            print(f"    midp:   {order_m}")


if __name__ == "__main__":
    main()
