import json
import re
import numpy as np
import plotly.graph_objects as go
from collections import defaultdict
import plotly.io as pio

# === 读取json ===
with open("set_run_summary.json", "r") as f:
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

# === 聚合数据 ===
groups = defaultdict(list)

for exp_name, record in data.items():
    if record["method"] == "tb_gfn" and record["size"] == "large" and record["fl"] == False:
        alpha = record["alpha_init"]
        if 0.4 <= alpha <= 0.9:
            steps = record["step"]
            values = record["forward_policy_entropy_eval"]
            groups[alpha].append(values)

# === 计算均值 ===
mean_results = {}
for alpha, runs in groups.items():
    runs = np.array(runs)  # shape: (n_runs, n_steps)
    mean_results[alpha] = runs.mean(axis=0)

steps = record["step"]  # 假设所有实验的 step 相同

# === Plotly 绘制折线图 ===
fig = go.Figure()

for alpha in sorted(mean_results.keys()):
    fig.add_trace(go.Scatter(
        x=steps,
        y=mean_results[alpha],
        mode="lines",
        name=fr"$\alpha={alpha}$"
    ))

fig.update_layout(
    width=800,   # 控制画布宽度
    height=500,  # 控制画布高度
    title=dict(
        text=r'$\alpha-\text{DB}$',
        font=dict(size=40, color="black"),   # 标题大且黑
        # 居中
        x=0.5,
        xanchor='center'
    ),
    xaxis=dict(
        title="Steps",
        title_font=dict(size=20, color="black"),  # x轴标题大且黑
        tickfont=dict(size=16, color="black")     # x轴刻度字体
    ),
    yaxis=dict(
        title="Average Forward Policy Entropy",
        title_font=dict(size=20, color="black"),  # y轴标题大且黑
        tickfont=dict(size=16, color="black")     # y轴刻度字体
    ),
    xaxis_title="Steps",
    yaxis_title="Average Forward Policy Entropy",
    legend=dict(
        # title="Alpha Init",
        x=0.79,   # 横向位置 (0=左, 1=右)
        y=0.01,   # 纵向位置 (0=底, 1=顶)
        xanchor="left",
        yanchor="bottom"
    ),
    template="plotly_white"
)
fig.update_xaxes(range=[0, 9000])
fig.write_html("test.html", include_mathjax="cdn")  # 或者 "require"
fig.show()
pio.write_image(fig, 'test4.pdf', width=800, height=500)