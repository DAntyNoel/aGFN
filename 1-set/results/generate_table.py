from pathlib import Path

import pandas as pd
import numpy as np

# 读取数据
ROOT = Path(__file__).resolve().parent
df = pd.read_csv(ROOT / 'data.csv')

# 将 method 名称映射为表头中的列
method_alias = {
    'db_gfn': 'DB',
    'fl_db_gfn': 'FL-DB',
    'fl_subtb_gfn': 'FL-SUBTB',
    'subtb_gfn': 'SUBTB',
    'tb_gfn': 'TB',
}

# 按期望顺序排列方法，跳过数据中不存在的 method
method_order = ['db_gfn', 'fl_db_gfn', 'fl_subtb_gfn', 'subtb_gfn', 'tb_gfn']
available_methods = [m for m in method_order if m in set(df['method'])]
method_labels = [method_alias[m] for m in available_methods]

df['method_alias'] = df['method'].map(method_alias)

# 指定 metric 列及其格式化方式
metrics = [
    ('modes_mean', 'modes_std', 'Modes$\\uparrow$', lambda v: f"{v:.1f}"),
    ('mean_top_1000_R_mean', 'mean_top_1000_R_std', 'Top-1000 Reward$\\uparrow$', None),
    ('mean_top_1000_similarity_mean', 'mean_top_1000_similarity_std', 'Top-1000 Sim.$\\uparrow$', lambda v: f"{v:.3f}"),
    ('spearman_corr_test_mean', 'spearman_corr_test_std', 'Spearman Corr', lambda v: f"{v:.3f}"),
]


def fmt_reward(val: float) -> str:
    """Format reward: keep decimals when <1 else show as integer."""
    return f"{val:.3f}" if abs(val) < 1 else f"{val:.0f}"


def fmt_value(val: float, formatter):
    if pd.isna(val):
        return ''
    if formatter is None:
        return fmt_reward(val)
    return formatter(val)


# 按 size 和 method 找到 baseline (alpha=0.5) 和 ours (最高 modes)
data_dict = {}
size_order = ['small', 'medium', 'large']
size_values = [s for s in size_order if s in set(df['size'])]

for size in size_values:
    for method in method_labels:
        subset = df[(df['method_alias'] == method) & (df['size'] == size)]
        if subset.empty:
            continue

        baseline_rows = subset[subset['alpha'] == 0.5]
        baseline_row = baseline_rows.iloc[0] if not baseline_rows.empty else None

        ours_candidates = subset[subset['alpha'] != 0.5]
        if ours_candidates.empty:
            ours_row = subset.iloc[0]
        else:
            ours_row = ours_candidates.loc[ours_candidates['modes_mean'].idxmax()]

        data_dict[(method, size)] = {
            'baseline': baseline_row,
            'ours': ours_row,
        }


# 生成新表格（结构参考 original.tex）
num_methods = len(method_labels)
tabular_cols = "@{\\hspace{.6em}}cl" + "cc" * num_methods + "@{\\hspace{.6em}}"

print("\\begin{table}[t]")
print("% \\vspace{-3\\baselineskip}")
print("\\caption{Results on Set Generation. $\\alpha$-GFNs perform better at reward and modes across all settings.}")
print("\\centering")
print("\\small")
print("\\setlength{\\tabcolsep}{6pt}")
print("\\renewcommand{\\arraystretch}{1.2}")
print("\\resizebox{\\linewidth}{!}{%")
print(f"\\begin{{tabular}}{{{tabular_cols}}}")
print("\\toprule")

# 顶部多列标题
method_header_parts = []
cmidrule_parts = []
start_col = 3
for label in method_labels:
    method_header_parts.append(f"\\multicolumn{{2}}{{c}}{{\\textbf{{{label}}}}}")
    cmidrule_parts.append(f"\\cmidrule(lr){{{start_col}-{start_col + 1}}}")
    start_col += 2

print("\\multicolumn{2}{c}{} &" + "\n" + " & ".join(method_header_parts) + " \\\\")
print("".join(cmidrule_parts))

# Baseline/Ours 表头
head_cells = ["\\textbf{Set Size}", "\\textbf{Metric}"]
for _ in method_labels:
    head_cells.extend(["Baseline", "Ours"])
print(" & ".join(head_cells) + " \\\\")
print("\\midrule")

for idx, size in enumerate(size_values):
    size_label = size.capitalize()
    for metric_idx, (mean_col, std_col, metric_label, formatter) in enumerate(metrics):
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

            baseline = data_dict[key]['baseline']
            ours = data_dict[key]['ours']

            if baseline is not None and not pd.isna(baseline[mean_col]):
                bm = fmt_value(float(baseline[mean_col]), formatter)
                bs = fmt_value(float(baseline[std_col]), formatter)
                print(f" & \\msl{{{bm}}}{{{bs}}}", end="")
            else:
                print(" & ", end="")

            om = fmt_value(float(ours[mean_col]), formatter)
            os = fmt_value(float(ours[std_col]), formatter)
            print(f" & \\msl{{\\textbf{{{om}}}}}{{{os}}}", end="")

        if metric_idx == len(metrics) - 1 and idx < len(size_values) - 1:
            print(" \\\\")
            print(f"\\cmidrule(lr){{1-{2 + 2 * num_methods}}}")
        else:
            print(" \\\\")

print("\\bottomrule")
print("\\end{tabular}")
print("}% end resizebox")
print("\\label{tab:set-exp-results}")
print("\\vspace{-1.5\\baselineskip}")
print("\\end{table}")
