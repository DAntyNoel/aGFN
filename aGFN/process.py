import pandas as pd
import ast
import numpy as np
from functools import partial


def parse_summary(s):
    try:
        return ast.literal_eval(s)
    except Exception:
        return {}

def format_val_d(m, s, decimals=2):
    """格式化 '均值 ± 标准差'
    
    Args:
        m: 均值
        s: 标准差
        decimals: 小数位数 (默认为2)
    
    Returns:
        格式化后的字符串
    """
    if pd.isna(m) or pd.isna(s):
        return "-"
    return f"{m:.{decimals}f} ± {s:.{decimals}f}"

def summary_bit(csv_path: str, k:int, target='modes', methods_order=None, T=True):
    # 读取csv
    df = pd.read_csv(csv_path)
    df = df[df["summary"].apply(lambda x: ast.literal_eval(x).get("step", None) == 49999)]
    df = df[df["summary"].apply(lambda x: ast.literal_eval(x).get("k", None) == k)]
    
    df["parsed"] = df["summary"].apply(parse_summary)
    df["alpha"] = df["parsed"].apply(lambda x: x.get("alpha_init", None))
    df["modes"] = df["parsed"].apply(lambda x: x.get("modes", None))
    df["objective"] = df["parsed"].apply(lambda x: x.get("objective", None))
    df["mean_top_100_R"] = df["parsed"].apply(lambda x: x.get("mean_top_100_R", None))
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
        latex.append("\\begin{table}[htbp]")
        latex.append("\\tiny")
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
                row.append(format_val_d(mean, std))
            latex.append(" & ".join(row) + " \\")
        
        latex.append("\\bottomrule")
        latex.append("\\end{tabular}")
        latex.append("\\end{table}")
        
        return "\n".join(latex)
    else:
        # 生成latex表格
        latex = []
        latex.append("\\begin{table}[h]")
        latex.append("\\tiny")
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
                row.append(format_val_d(mean, std))
            latex.append(" & ".join(row) + " \\")
        
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
        latex.append("\\begin{table}[htbp]")
        latex.append("\\tiny")
        latex.append("\\centering")
        latex.append(f"\\caption{{ {target.upper()} Results for $size={size}$}}")
        latex.append("\\begin{tabular}{c" + "c"*len(alphas) + "}")
        latex.append("\\toprule")
        
        # 表头
        header = "Method & " + " & ".join([f"{a:.1f}" for a in alphas]) + " \\\\"
        latex.append(header)
        latex.append("\\midrule")
        
        # 表格内容
        format_funcs = {}
        for m in methods:
            if target == "modes":
                format_funcs[m] = partial(format_val_d, decimals=1)
            elif target == "mean_top_1000_R":
                if size == 'small':
                    format_funcs[m] = partial(format_val_d, decimals=3)
                elif size == 'medium':
                    format_funcs[m] = partial(format_val_d, decimals=0)
                elif size == 'large':
                    format_funcs[m] = partial(format_val_d, decimals=0)
            elif target == 'spearman_corr_test':
                format_funcs[m] = partial(format_val_d, decimals=3)
            else:
                format_funcs[m] = partial(format_val_d, decimals=2)
        for m in methods:
            row = [m]
            format_val=format_funcs[m]
            for a in alphas:
                mean = table.loc[a, ("mean", m)] if (a, m) in grouped.set_index(["alpha","method"]).index else np.nan
                std = table.loc[a, ("std", m)] if (a, m) in grouped.set_index(["alpha","method"]).index else np.nan
                row.append(format_val(mean, std))
            latex.append(" & ".join(row) + " \\")
        
        latex.append("\\bottomrule")
        latex.append("\\end{tabular}")
        latex.append("\\end{table}")
        
        return "\n".join(latex)
    

def summary_mols(csv_path: str, target='modes', methods_order=None, T=True):
    # 读取csv
    df = pd.read_csv(csv_path)
    df = df[df["summary"].apply(lambda x: ast.literal_eval(x).get("step", None) == 49999)]
    
    df["parsed"] = df["summary"].apply(parse_summary)
    df["alpha"] = df["parsed"].apply(lambda x: x.get("alpha_init", None))
    df[target] = df["parsed"].apply(lambda x: x.get(target, None))
    df[target] = pd.to_numeric(df[target], errors='coerce')
    df["objective"] = df["parsed"].apply(lambda x: x.get("objective", None))
    df["mean_top_10_R"] = df["parsed"].apply(lambda x: x.get("top_10_avg_reward_eval", None))
    df["mean_top_100_R"] = df["parsed"].apply(lambda x: x.get("top_100_avg_reward_eval", None))
    df["mean_top_1000_R"] = df["parsed"].apply(lambda x: x.get("top_1000_avg_reward_eval", None))

    df["mean_top_10_similarity"] = df["parsed"].apply(lambda x: x.get("top_10_avg_similarity_eval", None))
    df["mean_top_100_similarity"] = df["parsed"].apply(lambda x: x.get("top_100_avg_similarity_eval", None))
    df["mean_top_1000_similarity"] = df["parsed"].apply(lambda x: x.get("top_1000_avg_similarity_eval", None))

    df["spearman_corr_test"] = df["parsed"].apply(lambda x: x.get("spearman_corr_test", None))
    df["modes"] = df["parsed"].apply(lambda x: x.get("num_modes_eval", None))
    num_cols = [
    'modes','mean_top_10_R','mean_top_100_R','mean_top_1000_R',
    'mean_top_10_similarity','mean_top_100_similarity','mean_top_1000_similarity',
    'spearman_corr_test'
    ]
    for c in set(num_cols).intersection(df.columns):
        df[c] = pd.to_numeric(df[c], errors='coerce')  # 或 .fillna(0)
    def get_method(summary):
        method = summary.get("objective", None)
        fl=summary.get('fl',False)
        if method == "db" and fl == True:
            return "fl-db"
        elif method=="subtb" and fl==True:
            return 'fl-subtb'
        return method

    df["objective"] = df["parsed"].apply(get_method)

    
    # 按 alpha, method 分组，计算均值和标准差
    grouped = df.groupby(["alpha", "objective"])[target].agg(["mean", "std"]).reset_index()

    # 生成透视表
    table = grouped.pivot(index="alpha", columns="objective", values=["mean", "std"])
    methods = methods_order if methods_order else sorted(df["objective"].unique())
    alphas = sorted(df["alpha"].unique())
    

    if T == True:
        # 生成latex表格
        latex = []
        latex.append("\\begin{table}[htbp]")
        latex.append('\\tiny')
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
                mean = table.loc[a, ("mean", m)] if (a, m) in grouped.set_index(["alpha","objective"]).index else np.nan
                std = table.loc[a, ("std", m)] if (a, m) in grouped.set_index(["alpha","objective"]).index else np.nan
                row.append(format_val_d(mean, std))
            latex.append(" & ".join(row) + " \\\\")
        
        latex.append("\\bottomrule")
        latex.append("\\end{tabular}")
        latex.append("\\end{table}")
        
        return "\n".join(latex)
    else:
        # 生成latex表格
        latex = []
        latex.append("\\begin{table}[htbp]")
        latex.append('\\tiny')
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
                row.append(format_val_d(mean, std))
            latex.append(" & ".join(row) + " \\\\")
        
        latex.append("\\bottomrule")
        latex.append("\\end{tabular}")
        latex.append("\\end{table}")
        
        return "\n".join(latex)

# === Plot: alpha comparison on large (mean_top_1000_R) ===

# def plot_alpha_compare_large(set_csv_path='set.csv', png_out='alpha_compare_large.png', pdf_out='alpha_compare_large.pdf',annotate='ratio',annotate_decimals=2):
#     """
#     Plotly grouped bar chart on large set comparing two alphas per objective
#       - db: 0.5 (Equal Weight) vs 0.9 (Mixed)
#       - fl-db: 0.5 (Equal Weight) vs 0.9 (Mixed)
#       - tb: 0.5 (Equal Weight) vs 0.7 (Mixed)
#     Bars show mean of mean_top_1000_R, with line error bars = std across runs.
#     Legend only: 'Equal Weight' vs 'Mixed'. No bar text or alpha labels on the plot.

#     Saves only static images (PNG and PDF) using plotly's static image export.
#     """
#     import plotly.graph_objects as go
#     import plotly.io as pio

#     # Load & filter
#     df = pd.read_csv(set_csv_path)
#     df = df[df['summary'].apply(lambda x: ast.literal_eval(x).get('step', None) == 9999)]
#     df = df[df['summary'].apply(lambda x: ast.literal_eval(x).get('size', None) == 'large')]

#     # Parse fields
#     df['parsed'] = df['summary'].apply(parse_summary)
#     df['alpha'] = df['parsed'].apply(lambda x: x.get('alpha_init', None))
#     # df['mean_top_1000_R'] = df['parsed'].apply(lambda x: x.get('mean_top_1000_R', None))
#     df['mean_top_1000_R'] = df['parsed'].apply(lambda x: x.get('mean_R', None))


#     def _get_method(summary):
#         method = summary.get('method', None)
#         if method == 'db_gfn' and summary.get('fl', False):
#             return 'fl_db_gfn'
#         return method

#     df['method'] = df['parsed'].apply(_get_method)
#     method_to_obj = {'db_gfn': 'db', 'fl_db_gfn': 'fl-db', 'tb_gfn': 'tb'}
#     df['objective'] = df['method'].map(method_to_obj)

#     # Keep only needed objectives & alphas
#     df = df[df['objective'].isin(['db', 'fl-db', 'tb'])]
#     chosen = {'db': {"eq": 0.5, "mix": 0.9}, 'fl-db': {"eq": 0.5, "mix": 0.9}, 'tb': {"eq": 0.5, "mix": 0.7}}
#     df = df[df.apply(lambda r: r['alpha'] in [chosen.get(r['objective'], {}).get('eq'), chosen.get(r['objective'], {}).get('mix')], axis=1)]

#     # Aggregate mean and std per (objective, alpha)
#     agg = df.groupby(['objective', 'alpha'])['mean_top_1000_R'].agg(['mean', 'std']).reset_index()

#     objectives = ['db', 'fl-db', 'tb']
#     eq_vals, eq_errs, mix_vals, mix_errs = [], [], [], []
#     for obj in objectives:
#         eq_alpha = chosen[obj]['eq']
#         mix_alpha = chosen[obj]['mix']
#         # Equal Weight
#         r_eq = agg[(agg['objective'] == obj) & (agg['alpha'] == eq_alpha)]
#         eq_vals.append(float(r_eq['mean'].iloc[0]) if len(r_eq) else np.nan)
#         eq_errs.append(float(r_eq['std'].iloc[0]) if len(r_eq) else np.nan)
#         # Mixed
#         r_mix = agg[(agg['objective'] == obj) & (agg['alpha'] == mix_alpha)]
#         mix_vals.append(float(r_mix['mean'].iloc[0]) if len(r_mix) else np.nan)
#         mix_errs.append(float(r_mix['std'].iloc[0]) if len(r_mix) else np.nan)

#     # Build Plotly figure
#     objectives=[x.upper() for x in objectives]  # Uppercase labels
#     fig = go.Figure()
#     fig.add_bar(
#         # name=r"$ \text{Vanilla GFN Objectives, } \alpha=0.5$',
#         name=r"$\text{Vanilla GFN Objectives, } \alpha=0.5$",
#         x=objectives,
#         y=eq_vals,
#         error_y=dict(type='data', array=eq_errs, visible=True),
#     )
#     fig.add_bar(
#         name=r"$\alpha \text{-GFN Objectives, }\alpha \in (0,1)$",
#         x=objectives,
#         y=mix_vals,
#         error_y=dict(type='data', array=mix_errs, visible=True),
#     )

#     fig.update_traces(width=0.4)   # 0~1 左右的比例，适当调小
#     n = len(objectives)
#     i_tb = objectives.index('TB')            # TB 在第几个
#     legend_x = (i_tb + 0.5) / n              # TB 组中心在 paper 坐标中的 x（0~1）

#     # 推荐的纵向位置（图内靠上），如有遮挡可把 0.92 调大/调小
#     legend_y = 0.8
#     fig.update_layout(
#         barmode='group',
#         # title='Alpha comparison on large set (mean_top_1000_R)',
#         xaxis_title='Objective',
#         yaxis_title='Average Reward',
#         legend_title_text='',
#         template='plotly_white',
#         bargap=0.2,        # 类别之间的空隙（越大柱越窄）
#         bargroupgap=0,    # 同一类别内两根柱之间的空隙（越大柱越窄）
#         legend=dict(
#             x=legend_x-0.18, y=legend_y,          # 放在图内左上角
#             xanchor='left', yanchor='top',
#             orientation='v',         # 竖排
#             bgcolor='rgba(255,255,255,0.6)',  # 半透明白底，读数不会被遮
#             bordercolor='rgba(0,0,0,0.2)',
#             borderwidth=1,
#             font=dict(size=10)       # 需要更紧凑可以再小一点
#         ),
#         # margin=dict(l=60, r=10, t=50, b=50)  # 稍微紧一点的页边距
#         margin=dict(l=28, r=6, t=8, b=26)  # 稍微紧一点的页边距

#     )
#     # === 独立的水平/竖直虚线括号 + 左侧倍率文本（不改你既有设定） ===
#     # 叠加线性 x 轴（仅用于精确放置，不影响原图）
#     n_groups = len(objectives)
#     fig.update_layout(
#         xaxis2=dict(
#             type='linear', overlaying='x',
#             range=[-0.5, n_groups - 0.5],
#             showgrid=False, ticks='', showticklabels=False, anchor='y'
#         )
#     )

#     # import plotly.graph_objects as go
#     idx = np.arange(n_groups)

#     # 与你的柱宽一致：update_traces(width=0.4) → 半宽 0.20
#     bar_half = 0.20        # 柱子半宽
#     delta    = 0.20        # 组内左右两柱相对组中心的偏移
#     cap_len  = 0.4        # 顶部短横线长度（可调，小则更像“一个点”）
#     vpad_x   = 0        # 竖线相对“低柱左边缘”再向左的距离（确保不压到柱/误差条）
#     gap_y    = 0.01 * np.nanmax([*eq_vals, *mix_vals])  # 横/竖线在 y 方向的分离缝

#     for i in range(n_groups):
#         if not (np.isfinite(eq_vals[i]) and np.isfinite(mix_vals[i])):
#             continue

#         # 若要对齐误差条顶端，把下两行改为 +eq_errs / +mix_errs
#         y_high = max(eq_vals[i], mix_vals[i])    # 高柱顶
#         y_low  = min(eq_vals[i], mix_vals[i])    # 低柱顶
#         y_cap = mix_vals[i]
#         sep = 0.001 * max(abs(eq_vals[i]), abs(mix_vals[i]), 1.0)

#         # 判定哪根是低柱，拿到它的“左边缘”
#         low_is_eq     = (eq_vals[i] <= mix_vals[i])
#         low_center_x  = (i - delta) if low_is_eq else (i + delta)
#         low_left_edge = low_center_x - bar_half

#         # 竖线放在低柱左边缘再向左一点；横线在 y_high+gap_y；竖线从 y_high-gap_y 到 y_low（两者留缝）
#         x_line      = low_left_edge - vpad_x
#         x_cap_left  = x_line 
#         x_cap_right = x_line + cap_len


#         # 顶部“帽檐”水平虚线（在竖线右侧或左侧按你现有 x_cap_left/x_cap_right）
#         fig.add_shape(
#             type='line', xref='x2', yref='y',
#             x0=x_cap_left, x1=x_cap_right,
#             y0=y_cap, y1=y_cap,                                # ← 这里与橙色柱顶持平
#             line=dict(color='rgba(0,0,0,0.65)', width=1.2, dash='dash')
#         )

#         # 竖直虚线（与横线分开，从 y_cap - sep 开始，指向低柱顶）
#         fig.add_shape(
#             type='line', xref='x2', yref='y',
#             x0=x_line, x1=x_line,
#             y0=y_cap - sep, y1=y_low,                          # ← 起点略低于帽檐，避免连成一体
#             line=dict(color='rgba(0,0,0,0.65)', width=1.2, dash='dash')
#         )

#         ratio_val = (mix_vals[i]/eq_vals[i]) if (np.isfinite(eq_vals[i]) and eq_vals[i] != 0) else np.nan
#         if np.isfinite(ratio_val):
#             ratio_txt = f"<b>{ratio_val:.{annotate_decimals}f}×</b>"  # ← 加粗
#             y_mid = 0.5 * (y_high + y_low)
#             fig.add_annotation(
#                 x=x_line + 0.37, y=y_mid, xref='x2', yref='y',
#                 text=ratio_txt, showarrow=False,
#                 xanchor='right', yanchor='middle',
#                 font=dict(size=14)  # ← 放大字号（可调 12~14）
#             )

#         # 去掉竖直网格线（以及 x 轴零线），保留 y 轴横向网格
#     fig.update_xaxes(showgrid=False, zeroline=False)
#     fig.update_yaxes(showgrid=True)  # 想更淡可加：gridcolor='rgba(0,0,0,0.08)'


#     # Save PNG and PDF only
#     try:
#         pio.write_image(fig, png_out,scale=2)
#     except Exception as e:
#         print(f"PNG export failed: {e}")
#     try:
#         pio.write_image(fig, pdf_out,scale=2)
#     except Exception as e:
#         print(f"PDF export failed: {e}")

def plot_alpha_compare_large(main_json_path='rebuttal_set_temp_old.json', json_path='set_run_summary_with_history.json', png_out='alpha_compare_large.png', pdf_out='alpha_compare_large.pdf', annotate='ratio', annotate_decimals=2, img_width=650, img_height=420):
    """
    Plotly grouped bar chart on large set comparing two alphas per objective
      - db: 0.5 (Equal Weight) vs 0.9 (Mixed)
      - fl-db: 0.5 (Equal Weight) vs 0.9 (Mixed)
      - tb: 0.5 (Equal Weight) vs 0.7 (Mixed)
      - subtb: 0.5 (Equal Weight) vs 0.9 (Mixed)  [新增]
    Bars show mean of mean_top_1000_R, with line error bars = std across runs.
    Legend only: 'Equal Weight' vs 'Mixed'. No bar text or alpha labels on the plot.

    Saves only static images (PNG and PDF) using plotly's static image export.
    """
    import ast
    import json
    import numpy as np
    import pandas as pd
    import plotly.graph_objects as go
    import plotly.io as pio

    # Load main objectives (db, fl-db, tb) from rebuttal_set_temp_old.json
    chosen = {'db': {"eq": 0.5, "mix": 0.9}, 'fl-db': {"eq": 0.5, "mix": 0.9}, 'tb': {"eq": 0.5, "mix": 0.7}}
    agg_list = []
    try:
        import re
        with open(main_json_path, 'r', encoding='utf-8') as f:
            main_json = json.load(f)

        rows = []
        for run_id, run_data in main_json.items():
            size_match = re.search(r'sz\(([^)]+)\)', run_id)
            size = size_match.group(1) if size_match else run_data.get('size')
            if size != 'large':
                continue  # keep parity with previous large-only CSV filter

            m_match = re.search(r'_m\(([^)]+)\)', run_id)
            method_val = m_match.group(1) if m_match else run_data.get('method')
            if method_val is None:
                continue

            mean_R = run_data.get('mean_R')
            if isinstance(mean_R, list) and len(mean_R) > 0:
                mean_R = mean_R[-1]

            rows.append({
                'run_id': run_id,
                'alpha': run_data.get('alpha_init'),
                'method': method_val,
                'mean_R': mean_R,
            })

        if rows:
            df_main = pd.DataFrame(rows)
            method_to_obj = {'db_gfn': 'db', 'fl_db_gfn': 'fl-db', 'tb_gfn': 'tb'}
            df_main['objective'] = df_main['method'].map(method_to_obj)
            df_main = df_main.dropna(subset=['objective', 'mean_R'])

            df_main = df_main[df_main.apply(lambda r: r['alpha'] in [chosen.get(r['objective'], {}).get('eq'), chosen.get(r['objective'], {}).get('mix')], axis=1)]

            agg_main = df_main.groupby(['objective', 'alpha'])['mean_R'].agg(['mean', 'std']).reset_index()
            agg_main.columns = ['objective', 'alpha', 'mean', 'std']
            agg_list.append(agg_main)
    except Exception as e:
        print(f"Warning: Could not load main JSON data: {e}")
        import traceback
        traceback.print_exc()

    # Load & process JSON data for new objectives (subtb, fl-subtb) parsed from run_id
    try:
        import re
        with open(json_path, 'r', encoding='utf-8') as f:
            json_data = json.load(f)
        
        json_list = []
        for run_id, run_data in json_data.items():
            # Extract size from run_id (e.g., "sz(large)" in key name)
            size_match = re.search(r'sz\(([^)]+)\)', run_id)
            size = size_match.group(1) if size_match else None

            # Extract method hint from run_id, fallback to payload
            method_hint = None
            m_match = re.search(r'_m\(([^)]+)\)', run_id)
            if m_match:
                method_hint = m_match.group(1)
            method_val = method_hint or run_data.get('method')

            # Size filter: baseline/subtb 使用 large；fl-subtb 允许 medium（或 large 以防有）
            if method_val == 'fl_subtb_gfn':
                if size not in ('medium', 'large'):
                    continue
            else:
                if size != 'large':
                    continue

            # Get the final mean_R value (could be array, take last value)
            mean_R = run_data.get('mean_R')
            if isinstance(mean_R, list) and len(mean_R) > 0:
                mean_R = mean_R[-1]  # Use last value from history

            json_list.append({
                'run_id': run_id,
                'alpha': run_data.get('alpha_init'),
                'method': method_val,
                'mean_R': mean_R,
            })
        
        if json_list:
            df_json = pd.DataFrame(json_list)
            # Map methods to objectives (fallback to parse from run_id)
            method_to_obj_json = {
                'subtb_gfn': 'subtb',
                'fl_subtb_gfn': 'fl-subtb',
            }
            df_json['objective'] = df_json['method'].map(method_to_obj_json)
            df_json.loc[df_json['objective'].isna(), 'objective'] = (
                df_json[df_json['objective'].isna()]['run_id']
                .str.extract(r'_m\(([^)]+)\)')[0]
                .map(method_to_obj_json)
            )
            df_json = df_json.dropna(subset=['objective', 'mean_R'])
            
            # Filter for chosen alphas
            df_json_filtered = df_json[df_json['alpha'].isin([0.5, 0.9])]
            
            # Add new objectives to chosen alphas (derive eq/mix from available alphas if needed)
            if not df_json_filtered.empty:
                agg_json = df_json_filtered.groupby(['objective', 'alpha'])['mean_R'].agg(['mean', 'std']).reset_index()
                agg_json.columns = ['objective', 'alpha', 'mean', 'std']

                # Derive chosen alphas per json objective
                for obj in agg_json['objective'].unique():
                    alphas = sorted(agg_json[agg_json['objective'] == obj]['alpha'].unique())
                    if len(alphas) >= 2:
                        eq_a, mix_a = alphas[0], alphas[-1]
                    elif len(alphas) == 1:
                        eq_a = mix_a = alphas[0]
                    else:
                        continue
                    chosen[obj] = {"eq": eq_a, "mix": mix_a}

                agg_list.append(agg_json)
                print(f"Added {len(df_json_filtered)} subtb entries to plot")
    except Exception as e:
        print(f"Warning: Could not load JSON data: {e}")
        import traceback
        traceback.print_exc()

    if agg_list:
        agg = pd.concat(agg_list, ignore_index=True)
    else:
        agg = pd.DataFrame(columns=['objective', 'alpha', 'mean', 'std'])

    # Only keep objectives that have both eq/mix data present
    preferred_order = ['db', 'subtb', 'tb', 'fl-db', 'fl-subtb']
    available = []
    for obj in preferred_order:
        if obj not in chosen:
            continue
        eq_alpha = chosen[obj]['eq']
        mix_alpha = chosen[obj]['mix']
        has_eq = not agg[(agg['objective'] == obj) & (agg['alpha'] == eq_alpha)].empty
        has_mix = not agg[(agg['objective'] == obj) & (agg['alpha'] == mix_alpha)].empty
        if has_eq and has_mix:
            available.append(obj)

    objectives = available
    
    print(f"Plotting objectives: {objectives}")
    print(f"Chosen alphas: {chosen}")
    
    eq_vals, eq_errs, mix_vals, mix_errs = [], [], [], []
    for obj in objectives:
        eq_alpha = chosen[obj]['eq']
        mix_alpha = chosen[obj]['mix']
        # Equal Weight
        r_eq = agg[(agg['objective'] == obj) & (agg['alpha'] == eq_alpha)]
        eq_vals.append(float(r_eq['mean'].iloc[0]) if len(r_eq) else np.nan)
        eq_errs.append(float(r_eq['std'].iloc[0]) if len(r_eq) else np.nan)
        # Mixed
        r_mix = agg[(agg['objective'] == obj) & (agg['alpha'] == mix_alpha)]
        mix_vals.append(float(r_mix['mean'].iloc[0]) if len(r_mix) else np.nan)
        mix_errs.append(float(r_mix['std'].iloc[0]) if len(r_mix) else np.nan)
        print(f"{obj}: eq={eq_vals[-1]:.2f}±{eq_errs[-1]:.2f}, mix={mix_vals[-1]:.2f}±{mix_errs[-1]:.2f}")

    # Build Plotly figure
    objectives=[x.upper() for x in objectives]  # Uppercase labels
    fig = go.Figure()
    fig.add_bar(
        # ← 改为 Unicode 文本
        name="Baseline (α = 0.5)",
        x=objectives,
        y=eq_vals,
        error_y=dict(type='data', array=eq_errs, visible=True),
    )
    fig.add_bar(
        # ← 改为 Unicode 文本
        name="Ours (α ∈ (0, 1))",
        x=objectives,
        y=mix_vals,
        error_y=dict(type='data', array=mix_errs, visible=True),
    )

    fig.update_traces(width=0.4)
    n = len(objectives)
    # 轴标题加粗 + 放大；Legend 放大字号
    fig.update_layout(
        barmode='group',
        xaxis_title="<b>Objective</b>",
        yaxis_title="<b>Average Reward</b>",
        legend_title_text='',
        template='plotly_white',
        bargap=0.2,
        bargroupgap=0,
        width=img_width,
        height=img_height,
        legend=dict(
            x=0.02, y=0.98,  # Move legend to top-left to avoid covering bars
            xanchor='left', yanchor='top',
            orientation='v',
            bgcolor='rgba(255,255,255,0.6)',
            bordercolor='rgba(0,0,0,0.2)',
            borderwidth=1,
            font=dict(size=22)   # ← 放大 legend 字号
        ),
        margin=dict(l=28, r=6, t=0, b=58)    )
    # 同时指定轴标题字体大小（粗体通过上面的 <b> 实现）
    # Enlarge axis title fonts
    fig.update_xaxes(title_font=dict(size=26),title_standoff=2, automargin=True,tickfont=dict(size=16))
    fig.update_yaxes(title_font=dict(size=26),automargin=True,tickfont=dict(size=16))

    # === 独立的水平/竖直虚线括号 + 左侧倍率文本（保留你的设定） ===
    n_groups = len(objectives)
    fig.update_layout(
        xaxis2=dict(
            type='linear', overlaying='x',
            range=[-0.5, n_groups - 0.5],
            showgrid=False, ticks='', showticklabels=False, anchor='y'
        )
    )

    idx = np.arange(n_groups)
    bar_half = 0.20
    delta    = 0.20
    cap_len  = 0.4
    vpad_x   = 0
    gap_y    = 0.01 * np.nanmax([*eq_vals, *mix_vals])

    for i in range(n_groups):
        if not (np.isfinite(eq_vals[i]) and np.isfinite(mix_vals[i])):
            continue

        y_high = max(eq_vals[i], mix_vals[i])
        y_low  = min(eq_vals[i], mix_vals[i])
        y_cap = mix_vals[i]
        sep = 0.001 * max(abs(eq_vals[i]), abs(mix_vals[i]), 1.0)

        low_is_eq     = (eq_vals[i] <= mix_vals[i])
        low_center_x  = (i - delta) if low_is_eq else (i + delta)
        low_left_edge = low_center_x - bar_half

        x_line      = low_left_edge - vpad_x
        x_cap_left  = x_line
        x_cap_right = x_line + cap_len

        fig.add_shape(
            type='line', xref='x2', yref='y',
            x0=x_cap_left, x1=x_cap_right,
            y0=y_cap, y1=y_cap,
            line=dict(color='rgba(0,0,0,0.65)', width=1.2, dash='dash')
        )
        fig.add_shape(
            type='line', xref='x2', yref='y',
            x0=x_line, x1=x_line,
            y0=y_cap - sep, y1=y_low,
            line=dict(color='rgba(0,0,0,0.65)', width=1.2, dash='dash')
        )

        ratio_val = (mix_vals[i]/eq_vals[i]) if (np.isfinite(eq_vals[i]) and eq_vals[i] != 0) else np.nan
        if np.isfinite(ratio_val):
            ratio_txt = f"<b>{ratio_val:.{annotate_decimals}f}×</b>"  # 使用 Unicode ×
            y_mid = y_cap + 3 * gap_y  # place slightly above the cap line
            fig.add_annotation(
                x=x_cap_right + 0.05, y=y_mid, xref='x2', yref='y',
                text=ratio_txt, showarrow=False,
                xanchor='right', yanchor='middle',
                font=dict(size=15)
            )

    fig.update_xaxes(showgrid=False, zeroline=False)
    fig.update_yaxes(showgrid=True)

    # Save PNG and PDF only
    try:
        pio.write_image(fig, png_out, scale=2, width=img_width, height=img_height)
    except Exception as e:
        print(f"PNG export failed: {e}")
    try:
        pio.write_image(fig, pdf_out, scale=2, width=img_width, height=img_height)
    except Exception as e:
        print(f"PDF export failed: {e}")



if __name__=="__main__":
    # size='large'
    # print(summary_set("set.csv", size=size, target='modes'))
    # print(summary_set("set.csv", size=size, target='mean_top_1000_R'))
    # print(summary_set("set.csv", size=size, target='spearman_corr_test'))


    # for k in [2,4,6,8,10]:
    #     for metric in ['modes','spearman_corr_test']:
    #         print(summary_bit("bit.csv", k=k, target=metric))
    metrics=['modes','mean_top_10_R','mean_top_100_R','mean_top_1000_R','mean_top_10_similarity','mean_top_100_similarity','mean_top_1000_similarity','spearman_corr_test']
    # metrics=['modes','mean_top_10_R','mean_top_100_R','mean_top_1000_R','mean_top_10_similarity','mean_top_100_similarity','mean_top_1000_similarity']
    # metrics=['spearman_corr_test']

    # metrics=['modes','mean_top_10_R','mean_top_100_R','mean_top_1000_R']

    # # metrics=['modes']
    # for metric in metrics:
    #     print(summary_mols('mols.csv',target=metric))

    # To produce the bar chart for large set (db: α=0.5/0.9, fl-db: α=0.5/0.9, tb: α=0.5/0.7, subtb: α=0.5/0.9):
    plot_alpha_compare_large('rebuttal_set_temp_old.json', json_path='set_run_summary_with_history.json', png_out='alpha_compare_large.png', pdf_out='alpha_compare_large.pdf')

