import json
import re
import numpy as np
import plotly.graph_objects as go
from collections import defaultdict
import plotly.io as pio
import os

# === 读取json ===
with open("set_run_summary_with_history.json", "r") as f:
    data = json.load(f)

# === 正则解析实验名中的参数 ===
def parse_exp_name(name):
    m_match = re.search(r"_m\((.*?)\)", name)
    a_match = re.search(r"_a\((.*?)\)", name)
    sz_match = re.search(r"_sz\((.*?)\)", name)
    return {
        "m": m_match.group(1) if m_match else None,
        "alpha_init": float(a_match.group(1)) if a_match else None,
        "size": sz_match.group(1) if sz_match else None,
    }


y_key="forward_policy_entropy_eval"
y_name="Avg Forward Policy Entropy"
y_key="spearman_corr_test"
y_name="Avg Spearman Correlation"
if not os.path.exists(y_key):
    os.makedirs(y_key)

# === 聚合数据 ===
for method,fl in [("tb_gfn", False), ("db_gfn", False), ("db_gfn", True)]:
    groups = defaultdict(list)
    
    print(f"\n=== Processing method={method}, fl={fl} ===")

    for exp_name, record in data.items():
        if record["method"] == method and record["size"] == 'large' and record["fl"] == fl:
            alpha = record["alpha_init"]
            if 0.4 <= alpha <= 0.9:
                steps = record["step"]
                values = record[y_key]
                groups[alpha].append(values)
    
    print(f"Found {len(groups)} alpha groups")
    for alpha, runs in groups.items():
        print(f"  alpha={alpha}: {len(runs)} runs")

    # === 计算均值 ===
    mean_results = {}
    for alpha, runs in groups.items():
        runs = np.array(runs)  # shape: (n_runs, n_steps)
        mean_results[alpha] = runs.mean(axis=0)

    print(f"mean_results has {len(mean_results)} entries")
    
    if len(mean_results) == 0:
        print("WARNING: No data found! Skipping this method.")
        continue

    # steps = record["step"]  # 假设所有实验的 step 相同
    # 1) 计算均值之后 & 使用前：把 steps 与曲线都截到 9k
    steps = np.asarray(record["step"])
    LIM = 9000
    mask = steps <= LIM

    for a in list(mean_results.keys()):
        mean_results[a] = np.asarray(mean_results[a])[mask]

    steps = steps[mask]  # 以后都用截断后的 steps

    print(f"After truncation to 9k: {len(steps)} steps, {len(mean_results)} alphas")

    # === Plotly 绘制折线图 ===
    fig = go.Figure()

    for alpha in sorted(mean_results.keys()):
        fig.add_trace(go.Scatter(
            x=steps,
            y=mean_results[alpha],
            mode="lines",
            name=f"α={alpha}"
        ))

    # —— 生成曲线略 —— #
    # 标题：用 Unicode 而不是 MathJax，保证能控制字号
    def method_title(method, fl):
        if method == "tb_gfn":
            return "TB"
        elif method == "db_gfn" and not fl:
            return "DB"
        else:
            return "FL-DB"

    title_text = method_title(method, fl)

    # 紧凑但不挤压的范围
    y_min = min(float(np.min(v)) for v in mean_results.values())
    y_max = max(float(np.max(v)) for v in mean_results.values())
    pad = 0.02 * (y_max - y_min if y_max > y_min else 1.0)
    y_range = [y_min - pad, y_max + pad]
    x_min, x_max = (min(steps), max(steps))

    fig.update_layout(
        width=800, height=600,
        # title=None,
        title=dict( text=title_text, font=dict(size=42, color="black"), x=0.5, xanchor="center", y=0.99, yanchor="top"), # 更靠近绘图区 pad=dict(t=0, b=0, l=0, r=0) # 去掉额外留白（关键
        margin=dict(l=80, r=22, t=42, b=70),
        legend=dict(
            x=0.99, y=0.01, xanchor="right", yanchor="bottom",
            orientation="v",
            font=dict(size=38),              # 图例字体
            bgcolor="rgba(255,255,255,0.6)"  # 可读性更好；想更紧凑可去掉
        ),
        template="plotly_white",
        paper_bgcolor="white",
        plot_bgcolor="white",
    )

    # X 轴：强制每 1000 一个刻度，并显示 1k/2k 格式
    fig.update_xaxes(
        range=[x_min, x_max],
        title=dict(text="Step", font=dict(size=40, color="black"), standoff=12),
        tickfont=dict(size=32, color="black"),
        ticks="outside",
        tickmode="linear",
        tick0=0,
        dtick=1000,
        tickformat="~s",
        automargin=True,
    )

    # Y 轴
    fig.update_yaxes(
        range=y_range,
        title=dict(text=y_name, font=dict(size=36, color="black"), standoff=12),
        tickfont=dict(size=32, color="black"),
        ticks="outside",
        automargin=True,
    )

    fig.update_layout(
        xaxis=dict(
            showline=True,
            linecolor="black",
            linewidth=1,
            mirror=True
        ),
        yaxis=dict(
            showline=True,
            linecolor="black",
            linewidth=1,
            mirror=True
        )
    )

    pio.write_image(fig, f'{y_key}/{method}-fl_{int(fl)}.pdf',
                    format='pdf', width=800, height=600, scale=2)
    print(f"Saved {y_key}/{method}-fl_{int(fl)}.pdf")
