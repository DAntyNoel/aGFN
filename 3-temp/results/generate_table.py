from pathlib import Path
import math

import pandas as pd

# Build a LaTeX table for the temperature-scaled results in data.csv
ROOT = Path(__file__).resolve().parent
DF_PATH = ROOT / "data.csv"
OUTPUT_TEX = ROOT / "new_table.tex"

df = pd.read_csv(DF_PATH)

# Map raw method names to table-friendly aliases
method_alias = {
    "db_gfn": "DB",
    "fl_db_gfn": "FL-DB",
    "fl_subtb_gfn": "FL-SUBTB",
    "subtb_gfn": "SUBTB",
    "tb_gfn": "TB",
}

method_order = ["db_gfn", "fl_db_gfn", "fl_subtb_gfn", "subtb_gfn", "tb_gfn"]
available_methods = [m for m in method_order if m in set(df["method"])]
method_labels = [method_alias[m] for m in available_methods]

df["method_alias"] = df["method"].map(method_alias)

# Metric columns: (mean, std, label, formatter, highlight_mean)
metrics = [
    ("modes_mean", "modes_std", "Modes$\\uparrow$", lambda v: f"{v:.1f}", True),
    ("mean_top_1000_R_mean", "mean_top_1000_R_std", "Top-1000 Reward$\\uparrow$", None, True),
    (
        "mean_top_1000_similarity_mean",
        "mean_top_1000_similarity_std",
        "Top-1000 Sim.$\\downarrow$",
        lambda v: f"{v:.3f}",
        True,
    ),
    ("spearman_corr_test_mean", "spearman_corr_test_std", "Spearman Corr", lambda v: f"{v:.3f}", False),
]


def fmt_reward(val: float) -> str:
    """Format reward: keep decimals when <1 else show as integer."""
    return f"{val:.3f}" if abs(val) < 1 else f"{val:.0f}"


def fmt_value(val: float, formatter):
    if pd.isna(val):
        return ""
    if formatter is None:
        return fmt_reward(val)
    return formatter(val)


def fmt_cell(val: float, std: float, formatter, highlight: bool) -> str:
    """Format mean/std pair with optional bold and scientific notation when large."""
    use_sci = abs(val) >= 100_000 and not pd.isna(val)
    if use_sci:
        exp = int(math.floor(math.log10(abs(val)))) if val != 0 else 0
        scale = 10 ** exp
        sval = val / scale
        sstd = std / scale
        if formatter is None:
            mean_str = f"{sval:.3g}"
            std_str = f"{sstd:.3g}"
        else:
            mean_str = fmt_value(sval, formatter)
            std_str = fmt_value(sstd, formatter)
        mean_fmt = f"\\textbf{{{mean_str}}}" if highlight and mean_str else mean_str
        base = f"$\\msl{{{mean_fmt}}}{{{std_str}}} \\times 10^{{{exp}}}$"
    else:
        mean_str = fmt_value(val, formatter)
        std_str = fmt_value(std, formatter)
        mean_fmt = f"\\textbf{{{mean_str}}}" if highlight and mean_str else mean_str
        base = f"\\msl{{{mean_fmt}}}{{{std_str}}}"

    return base


# Collect baseline (alpha=0.5) and best (max modes) per reward_temp/size/method
size_order = ["small", "medium", "large"]
reward_temps = sorted(df["reward_temp"].unique())

data_dict = {}
for temp in reward_temps:
    for size in size_order:
        for method in method_labels:
            subset = df[
                (df["method_alias"] == method)
                & (df["size"] == size)
                & (df["reward_temp"] == temp)
            ]
            if subset.empty:
                continue

            baseline_rows = subset[subset["alpha"] == 0.5]
            baseline_row = baseline_rows.iloc[0] if not baseline_rows.empty else None

            ours_candidates = subset[subset["alpha"] != 0.5]
            if ours_candidates.empty:
                ours_row = subset.iloc[0]
            else:
                ours_row = ours_candidates.loc[ours_candidates["modes_mean"].idxmax()]

            data_dict[(temp, size, method)] = {"baseline": baseline_row, "ours": ours_row}


# Generate LaTeX table
num_methods = len(method_labels)
tabular_cols = "@{\\hspace{.6em}}ccl" + "cc" * num_methods + "@{\\hspace{.6em}}"

lines = []
append = lines.append

append("\\begin{table}[t]")
append("% \\vspace{-3\\baselineskip}")
append("\\caption{Results on Set Generation across Temperature Scaling. $\\alpha$-GFNs show reward and modes improvements under varying reward temperatures.}")
append("\\centering")
append("\\small")
append("\\setlength{\\tabcolsep}{6pt}")
append("\\renewcommand{\\arraystretch}{1.2}")
append("\\resizebox{\\linewidth}{!}{%")
append(f"\\begin{{tabular}}{{{tabular_cols}}}")
append("\\toprule")

# Top multi-column headers
method_header_parts = []
cmidrule_parts = []
start_col = 4  # temp, size, metric occupy first three columns
for label in method_labels:
    method_header_parts.append(f"\\multicolumn{{2}}{{c}}{{\\textbf{{{label}}}}}")
    cmidrule_parts.append(f"\\cmidrule(lr){{{start_col}-{start_col + 1}}}")
    start_col += 2

append("\\multicolumn{3}{c}{} &" + "\n" + " & ".join(method_header_parts) + " \\\\")
append("".join(cmidrule_parts))

head_cells = ["\\textbf{Temp}", "\\textbf{Set Size}", "\\textbf{Metric}"]
for _ in method_labels:
    head_cells.extend(["Baseline", "Ours"])
append(" & ".join(head_cells) + " \\\\")
append("\\midrule")

total_cols = 3 + 2 * num_methods

for temp_idx, temp in enumerate(reward_temps):
    temp_subset_sizes = [s for s in size_order if any((temp, s, m) in data_dict for m in method_labels)]
    if not temp_subset_sizes:
        continue

    temp_rows = len(temp_subset_sizes) * len(metrics)
    temp_printed = False

    for size_idx, size in enumerate(temp_subset_sizes):
        size_label = size.capitalize()
        for metric_idx, (mean_col, std_col, metric_label, formatter, highlight_mean) in enumerate(metrics):
            temp_cell = f"\\multirow{{{temp_rows}}}{{*}}{{{temp}}}" if not temp_printed else ""
            if temp_cell:
                temp_printed = True

            size_cell = f"\\multirow{{{len(metrics)}}}{{*}}{{{size_label}}}" if metric_idx == 0 else ""

            row_cells = [temp_cell, size_cell, metric_label]

            for m in method_labels:
                key = (temp, size, m)
                if key not in data_dict:
                    row_cells.extend(["", ""])
                    continue

                baseline = data_dict[key]["baseline"]
                ours = data_dict[key]["ours"]

                # Determine which value to highlight (higher-is-better for all metrics here)
                baseline_mean = float(baseline[mean_col]) if baseline is not None else float("nan")
                ours_mean = float(ours[mean_col])
                best_mean = max(baseline_mean, ours_mean)

                if baseline is not None and not pd.isna(baseline_mean):
                    row_cells.append(
                        fmt_cell(
                            baseline_mean,
                            float(baseline[std_col]),
                            formatter,
                            highlight=highlight_mean and baseline_mean == best_mean,
                        )
                    )
                else:
                    row_cells.append("")

                row_cells.append(
                    fmt_cell(
                        ours_mean,
                        float(ours[std_col]),
                        formatter,
                        highlight=highlight_mean and (pd.isna(baseline_mean) or ours_mean == best_mean),
                    )
                )

            append(" & ".join(row_cells) + " \\\\")

            is_last_metric = metric_idx == len(metrics) - 1
            is_last_size = size_idx == len(temp_subset_sizes) - 1
            if is_last_metric and not is_last_size:
                append(f"\\cmidrule(lr){{2-{total_cols}}}")

    # Thick separator between different reward temperatures
    if temp_idx < len(reward_temps) - 1:
        append("\\midrule")

append("\\bottomrule")
append("\\end{tabular}")
append("}% end resizebox")
append("\\label{tab:set-exp-results-temp}")
append("\\vspace{-1.5\\baselineskip}")
append("\\end{table}")

# Write to file
OUTPUT_TEX.write_text("\n".join(lines), encoding="utf-8")

# Also echo to stdout for quick copy-paste
print("\n".join(lines))
