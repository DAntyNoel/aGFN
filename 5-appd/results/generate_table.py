from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
DF_PATH = ROOT / "data.csv"

df = pd.read_csv(DF_PATH)

method_alias = {
    "db_gfn": "DB",
    "fl_db_gfn": "FL-DB",
    "fl_subtb_gfn": "FL-SubTB($\\lambda$)",
    "subtb_gfn": "SubTB($\\lambda$)",
    "tb_gfn": "TB",
}

method_order = ["db_gfn", "fl_db_gfn", "subtb_gfn", "fl_subtb_gfn", "tb_gfn"]
available_methods = [m for m in method_order if m in set(df["method"])]
method_labels = [method_alias[m] for m in available_methods]

df["method_alias"] = df["method"].map(method_alias)

metrics = [
    ("modes_mean", "modes_std", "Modes$\\uparrow$", lambda v: f"{v:.1f}", True, False),
    ("mean_top_1000_R_mean", "mean_top_1000_R_std", "Top-1000 Reward$\\uparrow$", None, True, False),
    ("spearman_corr_test_mean", "spearman_corr_test_std", "Spearman Corr", lambda v: f"{v:.3f}", False, False),
]


def fmt_reward(val: float) -> str:
    return f"{val:.3f}" if abs(val) < 1 else f"{val:.0f}"


def fmt_value(val: float, formatter):
    if pd.isna(val):
        return ""
    if formatter is None:
        return fmt_reward(val)
    return formatter(val)


def fmt_cell(val: float, std: float, formatter, highlight: bool) -> str:
    mean_str = fmt_value(val, formatter)
    std_str = fmt_value(std, formatter)
    mean_fmt = f"\\textbf{{{mean_str}}}" if highlight and mean_str else mean_str
    return f"\\msl{{{mean_fmt}}}{{{std_str}}}"


size_order = ["small", "medium", "large"]
size_values = [s for s in size_order if s in set(df["size"])]

data_dict = {}
for size in size_values:
    for method in method_labels:
        subset = df[(df["method_alias"] == method) & (df["size"] == size)]
        if subset.empty:
            continue

        baseline_rows = subset[subset["alpha"] == 0.5]
        baseline_row = baseline_rows.iloc[0] if not baseline_rows.empty else None

        ours_candidates = subset[subset["alpha"] != 0.5]
        if ours_candidates.empty:
            ours_row = subset.iloc[0]
        else:
            ours_row = ours_candidates.loc[ours_candidates["modes_mean"].idxmax()]

        data_dict[(method, size)] = {"baseline": baseline_row, "ours": ours_row}


num_methods = len(method_labels)
tabular_cols = "@{\\hspace{.6em}}cl" + "cc" * num_methods + "@{\\hspace{.6em}}"

print("\\begin{table}[t]")
print("% \\vspace{-3\\baselineskip}")
print("\\caption{Results on Set Generation (Appendix).}")
print("\\centering")
print("\\small")
print("\\setlength{\\tabcolsep}{6pt}")
print("\\renewcommand{\\arraystretch}{1.2}")
print("\\resizebox{\\linewidth}{!}{%")
print(f"\\begin{{tabular}}{{{tabular_cols}}}")
print("\\toprule")

method_header_parts = []
cmidrule_parts = []
start_col = 3
for label in method_labels:
    method_header_parts.append(f"\\multicolumn{{2}}{{c}}{{\\textbf{{{label}}}}}")
    cmidrule_parts.append(f"\\cmidrule(lr){{{start_col}-{start_col + 1}}}")
    start_col += 2

print("\\multicolumn{2}{c}{} &" + "\n" + " & ".join(method_header_parts) + " \\\\")
print("".join(cmidrule_parts))

head_cells = ["\\textbf{Set Size}", "\\textbf{Metric}"]
for _ in method_labels:
    head_cells.extend(["Baseline", "Ours"])
print(" & ".join(head_cells) + " \\\\")
print("\\midrule")

for idx, size in enumerate(size_values):
    size_label = size.capitalize()
    for metric_idx, (mean_col, std_col, metric_label, formatter, highlight_mean, lower_is_better) in enumerate(metrics):
        row_prefix = f"\\multirow{{{len(metrics)}}}{{*}}{{{size_label}}} " if metric_idx == 0 else ""
        if metric_idx == 0:
            print(f"{row_prefix}\n& {metric_label}", end="")
        else:
            print(f"& {metric_label}", end="")

        for m in method_labels:
            key = (m, size)
            if key not in data_dict:
                print(" & &", end="")
                continue

            baseline = data_dict[key]["baseline"]
            ours = data_dict[key]["ours"]

            baseline_mean = float(baseline[mean_col]) if baseline is not None else float("nan")
            ours_mean = float(ours[mean_col])
            best_mean = min(baseline_mean, ours_mean) if lower_is_better else max(baseline_mean, ours_mean)

            if baseline is not None and not pd.isna(baseline_mean):
                bm = fmt_cell(baseline_mean, float(baseline[std_col]), formatter, highlight=highlight_mean and baseline_mean == best_mean)
                print(f" & {bm}", end="")
            else:
                print(" & ", end="")

            om = fmt_cell(ours_mean, float(ours[std_col]), formatter, highlight=highlight_mean and (pd.isna(baseline_mean) or ours_mean == best_mean))
            print(f" & {om}", end="")

        if metric_idx == len(metrics) - 1 and idx < len(size_values) - 1:
            print(" \\\\")
            print(f"\\cmidrule(lr){{1-{2 + 2 * num_methods}}}")
        else:
            print(" \\\\")

print("\\bottomrule")
print("\\end{tabular}")
print("}% end resizebox")
print("\\label{tab:set-exp-results-appd}")
print("\\vspace{-1.5\\baselineskip}")
print("\\end{table}")
