"""Format the rebuttal notebook's saved sensitivity results as LaTeX tables."""

from pathlib import Path

import pandas as pd


def _table(headers, rows, caption, label, align):
    lines = [
        r"\begin{table}[t]", r"\centering", r"\small",
        r"\setlength{\tabcolsep}{3.5pt}",
        rf"\caption{{{caption}}}", rf"\label{{{label}}}",
        r"\resizebox{\linewidth}{!}{%", rf"\begin{{tabular}}{{{align}}}",
        r"\toprule", " & ".join(headers) + r" \\", r"\midrule",
    ]
    previous_dataset = None
    for row in rows:
        if row[0] in {"SummEval", "HiTZ-BASSE", "HANNA", "All"}:
            if previous_dataset is not None and row[0] != previous_dataset:
                lines.append(r"\midrule")
            previous_dataset = row[0]
        lines.append(" & ".join(map(str, row)) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}%", "}", r"\end{table}"]
    return "\n".join(lines)


def _percent(value):
    return rf"{100 * float(value):.1f}\%"


def _dimension(value):
    return "5W1H" if str(value).casefold() == "5w1h" else str(value).lower()


def write_sensitivity_latex(output_dir):
    """Read saved CSV summaries; write a compact table and a complete table bundle."""
    root = Path(output_dir)
    compact = pd.read_csv(root / "reviewer3_compact_sensitivity_table.csv")
    overall = pd.read_csv(root / "reviewer3_sensitivity_stability_overall.csv").iloc[0]
    datasets = pd.read_csv(root / "reviewer3_sensitivity_stability_by_dataset.csv")
    dimensions = pd.read_csv(root / "reviewer3_sensitivity_stability_by_dimension.csv")
    oat = pd.read_csv(root / "reviewer3_sensitivity_one_at_a_time_summary.csv")
    stage2 = pd.read_csv(root / "reviewer3_stage2_mnsq_sensitivity.csv")
    ties = pd.read_csv(root / "reviewer2_strict_vs_mid_by_dataset.csv")
    n_settings = int(overall["n_dataset_dimensions"])
    n_configs = int(overall["n_configurations"])
    baseline = (
        r"Baseline: $\tau_H=\tau_L=0.80$, $\gamma=0.30$, and both infit and "
        r"outfit MNSQ in $[0.5,1.5]$."
    )
    summary_rows = [
        [str(value).replace("%", r"\%") for value in row]
        for row in compact.itertuples(index=False, name=None)
    ]
    summary = _table(
        ["Analysis", "Tested values", "Rank", "nSMF", "Risk"], summary_rows,
        rf"Operational sensitivity across {n_settings} dataset--dimension settings. "
        rf"The full grid has {n_configs} policies and reports the percentage of selections "
        r"unchanged from baseline. Other rows report numbers of settings with changed "
        r"selections, in the order of the tested values, while other parameters stay "
        r"at baseline. Matching non-selections count as unchanged. " + baseline + " "
        r"Grid frequencies are descriptive, not posterior probabilities; no models were refitted.",
        "tab:rebuttal-sensitivity-summary", "llccc",
    )

    stability_columns = [
        "rank_selection_stability", "pairwise_selection_stability",
        "risk_selection_stability", "decision_outcome_stability",
    ]
    dataset_rows = []
    for dataset in ["SummEval", "HiTZ-BASSE", "HANNA"]:
        row = datasets.set_index("dataset").loc[dataset]
        dataset_rows.append([dataset, int(row["n_dataset_dimensions"]),
                             *[_percent(row[c]) for c in stability_columns]])
    dataset_rows.append(["All", n_settings, *[_percent(overall[c]) for c in stability_columns]])
    by_dataset = _table(
        ["Dataset", "Dimensions", "Rank", "nSMF", "Risk", "Outcome"], dataset_rows,
        rf"Stability across the {n_configs}-policy factorial grid, relative to baseline. "
        r"Entries are percentages of dataset--dimension--policy comparisons with the same "
        r"selected judge (Rank, nSMF, Risk), or the same operational decision category "
        r"(Outcome). Matching non-selections count as unchanged. " + baseline,
        "tab:rebuttal-sensitivity-datasets", "llrrrr",
    )

    parameter_labels = {
        "tau_h": r"Human support $\tau_H$", "tau_l": r"Judge support $\tau_L$",
        "fsr_max": r"FSR tolerance $\gamma$", "mnsq_interval": "MNSQ interval",
    }
    oat_rows = []
    for parameter, label in parameter_labels.items():
        for row in oat.loc[oat["varied_parameter"].eq(parameter)].itertuples(index=False):
            value = str(row.parameter_value)
            if parameter == "mnsq_interval":
                low, high = map(float, value.split("-"))
                value = rf"$[{low:.2f},{high:.2f}]$"
            else:
                value = f"{float(value):.2f}"
            oat_rows.append([
                label, value, int(row.rank_decisions_changed), int(row.pairwise_decisions_changed),
                int(row.risk_decisions_changed), int(row.n_framework_refusals), int(row.n_risk_abstentions),
            ])
    one_at_a_time = _table(
        ["Varied parameter", "Value", "Rank changes", "nSMF changes", "Risk changes", "Refusals", "Abstentions"],
        oat_rows,
        rf"One-at-a-time sensitivity over {n_settings} dataset--dimension settings. "
        r"Change columns count selected-judge changes relative to baseline, including "
        r"transitions to or from non-selection. Refusals and risk abstentions are totals "
        r"under the tested policy, not changes. " + baseline,
        "tab:rebuttal-sensitivity-one-at-a-time", "llrrrrr",
    )

    dimension_rows = []
    for dataset in ["SummEval", "HiTZ-BASSE", "HANNA"]:
        for row in dimensions.loc[dimensions["dataset"].eq(dataset)].to_dict("records"):
            dimension_rows.append([dataset, _dimension(row["dimension"]),
                                   *[_percent(row[c]) for c in stability_columns]])
    by_dimension = _table(
        ["Dataset", "Dimension", "Rank", "nSMF", "Risk", "Outcome"], dimension_rows,
        rf"Per-dimension stability across the {n_configs}-policy grid. Entries are "
        r"percentages retaining the baseline selected judge or decision category; "
        r"matching non-selections count as unchanged. " + baseline,
        "tab:rebuttal-sensitivity-dimensions", "llrrrr",
    )

    intervals = sorted(set(zip(stage2["mnsq_low"], stage2["mnsq_high"])))
    gate_rows = []
    for dataset in ["SummEval", "HiTZ-BASSE", "HANNA"]:
        for dimension, group in stage2.loc[stage2["dataset"].eq(dataset)].groupby("dimension", sort=True):
            cells = []
            for low, high in intervals:
                row = group.loc[group["mnsq_low"].eq(low) & group["mnsq_high"].eq(high)]
                if len(row) != 1:
                    raise ValueError(f"Missing or duplicate MNSQ result: {dataset}/{dimension}/{low}/{high}")
                row = row.iloc[0]
                cells.append(f"{int(row['n_stage2_pass'])}/{int(row['n_judges'])}")
            gate_rows.append([dataset, _dimension(dimension), *cells])
    gates = _table(
        ["Dataset", "Dimension", *[rf"$[{low:.1f},{high:.1f}]$" for low, high in intervals]],
        gate_rows,
        r"Stage~2 sensitivity to the admissible MNSQ interval. Each entry is the number "
        r"of passing judges over the number evaluated. Both infit and outfit must lie "
        r"in the stated closed interval.",
        "tab:rebuttal-sensitivity-mnsq", "llccc",
    )

    tie_rows = []
    for dataset in ["SummEval", "HiTZ-BASSE", "HANNA"]:
        row = ties.set_index("dataset").loc[dataset]
        n = int(row["n_dataset_dimensions"])
        tie_rows.append([dataset,
                         f"{int(row['settings_where_original_minimizers_retained'])}/{n}",
                         f"{int(row['settings_with_exactly_same_argmin_set'])}/{n}",
                         f"{int(row['risk_outcomes_changed'])}/{n}",
                         f"{int(row['selection_to_abstention'])}/{n}"])
    tie_table = _table(
        ["Dataset", "Original minima retained", "Same minimum set", "Risk changes", "Select to abstain"],
        tie_rows,
        r"Sensitivity to strict versus mid-dominance. Counts are over dataset dimensions. "
        r"The first two columns compare the judges minimizing FSR; the last two compare "
        r"operational risk-selection outcomes. Mid-dominance assigns half of exact tie "
        r"mass to each direction; thresholds and gates are otherwise unchanged.",
        "tab:rebuttal-sensitivity-ties", "lcccc",
    )

    preamble = "% Requires \\usepackage{booktabs,graphicx}.\n% Generated from saved rebuttal CSV summaries.\n"
    summary_path = root / "rebuttal_sensitivity_summary.tex"
    summary_path.write_text(preamble + summary + "\n", encoding="utf-8")
    bundle_path = root / "rebuttal_sensitivity_tables.tex"
    bundle_path.write_text(
        preamble + "\n\n".join([summary, by_dataset, one_at_a_time, by_dimension, gates, tie_table]) + "\n",
        encoding="utf-8",
    )
    return summary_path, bundle_path
