import pandas as pd


def fmt_sci(val: float) -> str:
    return f"{val:.2e}"


def build_table(df: pd.DataFrame) -> None:
    method_alias = {
        "db": "DB",
        "subtb": "SubTB($\\lambda$)",
        "tb": "TB",
    }

    method_order = ["db", "subtb", "tb"]
    available_methods = [m for m in method_order if m in set(df["method"])]
    method_labels = [method_alias[m] for m in available_methods]

    df["method_alias"] = df["method"].map(method_alias)

    data_dict = {}
    for k in sorted(df["k"].unique()):
        for method in method_labels:
            subset = df[(df["method_alias"] == method) & (df["k"] == k)]
            if subset.empty:
                continue

            baseline_rows = subset[subset["alpha"] == 0.5]
            baseline_row = baseline_rows.iloc[0] if not baseline_rows.empty else None

            ours_rows = subset[subset["alpha"] != 0.5]
            if not ours_rows.empty:
                ours_row = ours_rows.loc[ours_rows["modes_mean"].idxmax()]
            elif baseline_row is not None:
                ours_row = baseline_row
            else:
                ours_row = subset.iloc[0]

            data_dict[(method, k)] = {"baseline": baseline_row, "ours": ours_row}

    print("\\begin{table}[htbp]")
    print("\\caption{Results on Bit Sequence Generation (Appendix).}")
    print("\\centering")
    print("\\small")
    print("\\setlength{\\tabcolsep}{6pt}")
    print("\\renewcommand{\\arraystretch}{1.2}")
    print("\\resizebox{0.99\\linewidth}{!}{%")

    num_methods = len(method_labels)
    tabular_cols = "ll" + "cc" * num_methods
    print(f"\\begin{{tabular}}{{{tabular_cols}}}")
    print("\\toprule")

    method_header_parts = [f"\\multicolumn{{2}}{{c}}{{\\textbf{{{label}}}}}" for label in method_labels]
    cmidrule_parts = []
    start_col = 3
    for _ in method_labels:
        cmidrule_parts.append(f"\\cmidrule(lr){{{start_col}-{start_col + 1}}}")
        start_col += 2

    print("\\multicolumn{1}{c}{} & \\multicolumn{1}{c}{} & " + " & ".join(method_header_parts) + " \\\\")
    print("".join(cmidrule_parts))
    head_cells = ["\\textbf{k}", "\\textbf{Metric}"]
    for _ in method_labels:
        head_cells.extend(["Baseline", "Ours"])
    print(" & ".join(head_cells) + " \\\\")
    print("\\midrule")

    k_values = sorted(df["k"].unique())

    for idx, k in enumerate(k_values):
        # Modes
        print(f"\\multirow{{3}}{{*}}{{{k}}} & Modes", end="")
        for method in method_labels:
            key = (method, k)
            if key in data_dict:
                baseline = data_dict[key]["baseline"]
                ours = data_dict[key]["ours"]
                if baseline is not None:
                    m = float(baseline["modes_mean"])
                    s = float(baseline["modes_std"])
                    print(f" & \\msl{{{m:.2f}}}{{{s:.2f}}}", end="")
                m = float(ours["modes_mean"])
                s = float(ours["modes_std"])
                print(f" & \\msl{{\\textbf{{{m:.2f}}}}}{{{s:.2f}}}", end="")
        print(" \\\\")

        # Spearman
        print("& Spearman", end="")
        for method in method_labels:
            key = (method, k)
            if key in data_dict:
                baseline = data_dict[key]["baseline"]
                ours = data_dict[key]["ours"]
                if baseline is not None:
                    m = float(baseline["spearman_corr_test_mean"])
                    s = float(baseline["spearman_corr_test_std"])
                    print(f" & \\msl{{{m:.2f}}}{{{s:.2f}}}", end="")
                m = float(ours["spearman_corr_test_mean"])
                s = float(ours["spearman_corr_test_std"])
                print(f" & \\msl{{\\textbf{{{m:.2f}}}}}{{{s:.2f}}}", end="")
        print(" \\\\")

        # MeanTop1kR
        print("& MeanTop1kR", end="")
        for method in method_labels:
            key = (method, k)
            if key in data_dict:
                baseline = data_dict[key]["baseline"]
                ours = data_dict[key]["ours"]
                if baseline is not None:
                    m = float(baseline["mean_top_1000_R_mean"])
                    s = float(baseline["mean_top_1000_R_std"])
                    print(f" & \\msl{{{fmt_sci(m)}}}{{{fmt_sci(s)}}}", end="")
                m = float(ours["mean_top_1000_R_mean"])
                s = float(ours["mean_top_1000_R_std"])
                print(f" & \\msl{{\\textbf{{{fmt_sci(m)}}}}}{{{fmt_sci(s)}}}", end="")

        if idx < len(k_values) - 1:
            print(" \\\\")
            print("\\midrule")
        else:
            print(" \\\\")

    print("\\bottomrule")
    print("\\end{tabular}")
    print("}% end resizebox")
    print("\\label{tab:bit-exp-results-appd}")
    print("% \\vspace{-1.5\\baselineskip}")
    print("\\end{table}")


if __name__ == "__main__":
    df = pd.read_csv("data.csv")
    build_table(df)
