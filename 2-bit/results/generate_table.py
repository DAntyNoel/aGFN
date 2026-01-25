import pandas as pd


def build_table(df: pd.DataFrame) -> None:
    # 构造 baseline（alpha=0.5）和 ours（非 0.5，若缺失则回退到 baseline 或首行）
    data_dict = {}
    for k in sorted(df['k'].unique()):
        for method in df['method'].unique():
            subset = df[(df['method'] == method) & (df['k'] == k)]
            if subset.empty:
                continue

            baseline_rows = subset[subset['alpha'] == 0.5]
            baseline_row = baseline_rows.iloc[0] if not baseline_rows.empty else None

            ours_rows = subset[subset['alpha'] != 0.5]
            if not ours_rows.empty:
                ours_row = ours_rows.iloc[0]
            elif baseline_row is not None:
                ours_row = baseline_row
            else:
                ours_row = subset.iloc[0]

            data_dict[(method, k)] = {'baseline': baseline_row, 'ours': ours_row}

    print("\\begin{table}[htbp]")
    print("\\caption{Results on Bit Sequence Generation. In terms of number of modes on average, $\\alpha$-GFN objectives outperform vanilla GFlowNet objectives across 87\\% task settings.}")
    print("\\centering")
    print("\\small")
    print("\\setlength{\\tabcolsep}{6pt}")
    print("\\renewcommand{\\arraystretch}{1.2}")
    print("\\resizebox{0.99\\linewidth}{!}{%")
    print("\\begin{tabular}{llcccccccccc}")
    print("\\toprule")
    print("\\multicolumn{1}{c}{} & \\multicolumn{1}{c}{} & \\multicolumn{2}{c}{\\textbf{DB}} & \\multicolumn{2}{c}{\\textbf{FL-DB}} & \\multicolumn{2}{c}{\\textbf{SubTB($\\lambda$)}} & \\multicolumn{2}{c}{\\textbf{FL-SubTB($\\lambda$)}} & \\multicolumn{2}{c}{\\textbf{TB}} \\\\")
    print("\\cmidrule(lr){3-4}\\cmidrule(lr){5-6}\\cmidrule(lr){7-8}\\cmidrule(lr){9-10}\\cmidrule(lr){11-12}")
    print("\\textbf{k} & \\textbf{Metric} & Baseline & Ours & Baseline & Ours & Baseline & Ours & Baseline & Ours & Baseline & Ours \\\\")
    print("\\midrule")

    methods_order = ['db', 'fl-db', 'subtb', 'fl-subtb', 'tb']
    k_values = sorted(df['k'].unique())

    for idx, k in enumerate(k_values):
        # Modes
        print(f"\\multirow{{3}}{{*}}{{{k}}} &  Modes", end="")
        for method in methods_order:
            key = (method, k)
            if key in data_dict:
                baseline = data_dict[key]['baseline']
                ours = data_dict[key]['ours']
                if baseline is not None:
                    m = float(baseline['modes_mean'])
                    s = float(baseline['modes_std'])
                    print(f" & \\msl{{{m:.2f}}}{{{s:.2f}}}", end="")
                m = float(ours['modes_mean'])
                s = float(ours['modes_std'])
                print(f" & \\msl{{\\textbf{{{m:.2f}}}}}{{{s:.2f}}}", end="")
        print(" \\\\")

        # Spearman
        print("&  Spearman", end="")
        for method in methods_order:
            key = (method, k)
            if key in data_dict:
                baseline = data_dict[key]['baseline']
                ours = data_dict[key]['ours']
                if baseline is not None:
                    m = float(baseline['spearman_corr_test_mean'])
                    s = float(baseline['spearman_corr_test_std'])
                    print(f" & \\msl{{{m:.2f}}}{{{s:.2f}}}", end="")
                m = float(ours['spearman_corr_test_mean'])
                s = float(ours['spearman_corr_test_std'])
                print(f" & \\msl{{{m:.2f}}}{{{s:.2f}}}", end="")
        print(" \\\\")

        # mean_top_1000_R
        print("&  MeanTop1kR", end="")
        for method in methods_order:
            key = (method, k)
            if key in data_dict:
                baseline = data_dict[key]['baseline']
                ours = data_dict[key]['ours']
                if baseline is not None:
                    m = float(baseline['mean_top_1000_R_mean'])
                    s = float(baseline['mean_top_1000_R_std'])
                    print(f" & \\msl{{{m:.2e}}}{{{s:.2e}}}", end="")
                m = float(ours['mean_top_1000_R_mean'])
                s = float(ours['mean_top_1000_R_std'])
                print(f" & \\msl{{{m:.2e}}}{{{s:.2e}}}", end="")

        if idx < len(k_values) - 1:
            print(" \\\\")
            print("\\midrule")
        else:
            print(" \\\\")

    print("\\bottomrule")
    print("\\end{tabular}")
    print("}% end resizebox")
    print("\\label{tab:bit-exp-results}")
    print("% \\vspace{-1.5\\baselineskip}")
    print("\\end{table}")


if __name__ == "__main__":
    df = pd.read_csv('data.csv')
    build_table(df)
