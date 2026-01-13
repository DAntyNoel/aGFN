import pandas as pd
import ast
import numpy as np

def parse_summary(s):
    try:
        return ast.literal_eval(s)
    except Exception:
        return {}
# 格式化 "均值 ± 标准差"
def format_val(m, s, b):
    if pd.isna(m): 
        return "-"
    if m > b:
        return f"\\msl{{\\textbf{{{m:.2f}}}}}{{{s:.2f}}}"
    return f"\\msl{{{m:.2f}}}{{{s:.2f}}}"

def summary_bit(csv_path: str, k:int, target='modes', methods_order=None, T=True):
    # 读取csv
    df = pd.read_csv(csv_path)
    df = df[df["summary"].apply(lambda x: ast.literal_eval(x).get("step", None) == 49999)]
    df = df[df["summary"].apply(lambda x: ast.literal_eval(x).get("k", None) == k)]
    
    df["parsed"] = df["summary"].apply(parse_summary)
    df["alpha"] = df["parsed"].apply(lambda x: x.get("alpha_init", None))
    df["modes"] = df["parsed"].apply(lambda x: x.get("modes", None))
    df["objective"] = df["parsed"].apply(lambda x: x.get("objective", None))
    df["mean_top_1000_R"] = df["parsed"].apply(lambda x: x.get("mean_top_1000_R", None))
    df["spearman_corr_test"] = df["parsed"].apply(lambda x: x.get("spearman_corr_test", None))
    
    # 按 alpha, method 分组，计算均值和标准差
    grouped = df.groupby(["alpha", "objective"])[target].agg(["mean", "std"]).reset_index()

    # 生成透视表
    table = grouped.pivot(index="alpha", columns="objective", values=["mean", "std"])
    methods = methods_order if methods_order else sorted(df["objective"].unique())
    alphas = sorted(df["alpha"].unique())
    

    if T == True:
        # 生成latex表格
        latex = []
        latex.append("\\begin{table}[h]")
        latex.append("\\centering")
        latex.append(f"\\caption{{ {target.upper()} Results for $k={k}$}}")
        latex.append("\\begin{tabular}{c" + "c"*len(alphas) + "}")
        latex.append("\\toprule")
        
        # 表头
        header = "Method & " + " & ".join([f"{a:.1f}" for a in alphas]) + " \\\\"
        latex.append(header)
        latex.append("\\midrule")
        
        # 表格内容
        for m in methods:
            row = [m]
            for a in alphas:
                mean = table.loc[a, ("mean", m)] if (a, m) in grouped.set_index(["alpha","objective"]).index else np.nan
                std = table.loc[a, ("std", m)] if (a, m) in grouped.set_index(["alpha","objective"]).index else np.nan
                row.append(format_val(mean, std))
            latex.append(" & ".join(row) + " \\\\")
        
        latex.append("\\bottomrule")
        latex.append("\\end{tabular}")
        latex.append("\\end{table}")
        
        return "\n".join(latex)
    else:
        # 生成latex表格
        latex = []
        latex.append("\\begin{table}[h]")
        latex.append("\\centering")
        latex.append("\\begin{tabular}{c" + "c"*len(methods) + "}")
        latex.append("\\toprule")
        
        # 表头
        header = "Alpha & " + " & ".join(methods) + " \\\\"
        latex.append(header)
        latex.append("\\midrule")

        
        # 表格内容
        for a in alphas:
            row = [f"{a:.1f}"]
            for m in methods:
                mean = table.loc[a, ("mean", m)] if (a, m) in grouped.set_index(["alpha","objective"]).index else np.nan
                std = table.loc[a, ("std", m)] if (a, m) in grouped.set_index(["alpha","objective"]).index else np.nan
                row.append(format_val(mean, std))
            latex.append(" & ".join(row) + " \\\\")
        
        latex.append("\\bottomrule")
        latex.append("\\end{tabular}")
        latex.append("\\end{table}")
        
        return "\n".join(latex)

def summary_set(csv_path: str, size: str, target='modes', methods_order=['db_gfn', 'fl_db_gfn', 'tb_gfn']):
    # 读取csv
    df = pd.read_csv(csv_path)
    df = df[df["summary"].apply(lambda x: ast.literal_eval(x).get("step", None) == 9999)]
    df = df[df["summary"].apply(lambda x: ast.literal_eval(x).get("size", None) == size)]

    df["parsed"] = df["summary"].apply(parse_summary)
    df["alpha"] = df["parsed"].apply(lambda x: x.get("alpha_init", None))
    df["modes"] = df["parsed"].apply(lambda x: x.get("modes", None))
    df["mean_top_1000_R"] = df["parsed"].apply(lambda x: x.get("mean_top_1000_R", None))
    df["spearman_corr_test"] = df["parsed"].apply(lambda x: x.get("spearman_corr_test", None))
    def get_method(summary):
        method = summary.get("method", None)
        if method == "db_gfn" and summary['fl'] == True:
            return "fl_db_gfn"
        return method

    df["method"] = df["parsed"].apply(get_method)
    
    # 按 alpha, method 分组，计算均值和标准差
    grouped = df.groupby(["alpha", "method"])[target].agg(["mean", "std"]).reset_index()
    # 生成透视表
    table = grouped.pivot(index="alpha", columns="method", values=["mean", "std"])
    methods = methods_order if methods_order else sorted(df["method"].unique())
    alphas = sorted(df["alpha"].unique())

    if True:
        # 生成latex表格
        latex = []
        latex.append("\\begin{table}[h]")
        latex.append("\\centering")
        latex.append(f"\\caption{{ {target.upper()} Results for $size={size}$}}")
        latex.append("\\begin{tabular}{c" + "c"*len(alphas) + "}")
        latex.append("\\toprule")
        
        # 表头
        header = "Method & " + " & ".join([f"{a:.1f}" for a in alphas]) + " \\\\"
        latex.append(header)
        latex.append("\\midrule")
        
        # 表格内容
        for m in methods:
            row = [m]
            for a in alphas:
                mean = table.loc[a, ("mean", m)] if (a, m) in grouped.set_index(["alpha","method"]).index else np.nan
                std = table.loc[a, ("std", m)] if (a, m) in grouped.set_index(["alpha","method"]).index else np.nan
                row.append(format_val(mean, std))
            latex.append(" & ".join(row) + " \\\\")
        
        latex.append("\\bottomrule")
        latex.append("\\end{tabular}")
        latex.append("\\end{table}")
        
        return "\n".join(latex)

def summary_mols(csv_path: str, target='modes', methods_order=None, fl=False, T=True):
    # 读取csv
    df = pd.read_csv(csv_path)
    def filter(x):
        d = ast.literal_eval(x)
        if d.get("step", None) != 49999:
            return False
        if d.get("random_action_prob", None) != 0.05:
            return False
        if d.get("use_exp_weight_decay", None) != False:
            return False
        # if d.get("fl", None) != fl:
        #     return False
        if d.get(target, None) == 'NaN':
            return False
        return True
    def get_objective(x):
        if x['fl']:
            return 'fl_' + x['objective']
        return x.get('objective', None)
    df = df[df["summary"].apply(filter)]
    
    df["parsed"] = df["summary"].apply(parse_summary)
    df["alpha"] = df["parsed"].apply(lambda x: x.get("alpha_init", None))
    df[target] = df["parsed"].apply(lambda x: x.get(target, None))
    df["objective"] = df["parsed"].apply(get_objective)
    df["mean_top_1000_R"] = df["parsed"].apply(lambda x: x.get("mean_top_1000_R", None))
    df["spearman_corr_test"] = df["parsed"].apply(lambda x: x.get("spearman_corr_test", None))
    
    # 按 alpha, method 分组，计算均值和标准差
    grouped = df.groupby(["alpha", "objective"])[target].agg(["mean", "std"]).reset_index()

    # 生成透视表
    table = grouped.pivot(index="alpha", columns="objective", values=["mean", "std"])
    methods = methods_order if methods_order else sorted(df["objective"].unique())
    alphas = sorted(df["alpha"].unique())
    

    if T == True:
        # 生成latex表格
        latex = []
        latex.append("\\begin{table}[h]")
        latex.append("\\centering")
        latex.append(f"\\caption{{ {target.upper()} Results}}")
        latex.append("\\begin{tabular}{c" + "c"*len(alphas) + "}")
        latex.append("\\toprule")
        
        # 表头
        header = "Method & " + " & ".join([f"{a:.1f}" for a in alphas]) + " \\\\"
        latex.append(header)
        latex.append("\\midrule")
        
        # 表格内容
        for m in methods:
            row = [m]
            for a in alphas:
                # if a not in [0.8,0.9]:
                #     continue
                base_mean = table.loc[0.5, ("mean", m)] if (0.5, m) in grouped.set_index(["alpha","objective"]).index else np.nan
                mean = table.loc[a, ("mean", m)] if (a, m) in grouped.set_index(["alpha","objective"]).index else np.nan
                std = table.loc[a, ("std", m)] if (a, m) in grouped.set_index(["alpha","objective"]).index else np.nan
                row.append(format_val(mean, std, base_mean))
            latex.append(" & ".join(row) + " \\\\")
        
        latex.append("\\bottomrule")
        latex.append("\\end{tabular}")
        latex.append("\\end{table}")
        
        return "\n".join(latex)
    else:
        # 生成latex表格
        latex = []
        latex.append("\\begin{table}[h]")
        latex.append("\\centering")
        latex.append("\\begin{tabular}{c" + "c"*len(methods) + "}")
        latex.append("\\toprule")
        
        # 表头
        header = "Alpha & " + " & ".join(methods) + " \\\\"
        latex.append(header)
        latex.append("\\midrule")

        
        # 表格内容
        for a in alphas:
            row = [f"{a:.1f}"]
            for m in methods:
                mean = table.loc[a, ("mean", m)] if (a, m) in grouped.set_index(["alpha","objective"]).index else np.nan
                std = table.loc[a, ("std", m)] if (a, m) in grouped.set_index(["alpha","objective"]).index else np.nan
                row.append(format_val(mean, std))
            latex.append(" & ".join(row) + " \\\\")
        
        latex.append("\\bottomrule")
        latex.append("\\end{tabular}")
        latex.append("\\end{table}")
        
        return "\n".join(latex)



# print(summary_set("set.csv", size='small', target='modes'))
# print(summary_bit("bit.csv", k=6, target='modes'))
print(summary_mols("mols.csv", target='num_modes_eval'))

