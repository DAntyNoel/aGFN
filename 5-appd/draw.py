import json
import os
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
from collections import defaultdict

def build_image(
        steps,
        groups,
        ours_alpha,
        title,
        ylabel,
        metric,
        file_name,
        x_range=[0, 50000],
        ticks=[0, 10000, 20000, 30000, 40000],
        tick_labels=["0", "10k", "20k", "30k", "40k"],
        otick=49900,
        otick_label="50k",
        width=800,
        height=600,
        showline=True,
        save_html=False,
        show_pic=False
    ):
    '''
    steps: list of int, (n_steps,)
    groups[alpha]: (n_runs, n_steps)
    '''
    # === 创建图 ===
    fig = go.Figure()
    # === 绘制均值和方差 ===
    for alpha in [0.5, ours_alpha]:
        runs = np.array(groups[alpha])  # shape: (n_runs, n_steps)
        mean_curve = runs.mean(axis=0)
        std_curve = runs.std(axis=0)

        # 阴影区域 (mean ± std)
        fig.add_trace(
            go.Scatter(
                x=steps.tolist() + steps[::-1].tolist(),
                y=(mean_curve + std_curve).tolist() + (mean_curve - std_curve)[::-1].tolist(),
                fill="toself",
                fillcolor="rgba(0,100,80,0.2)" if alpha == 0.5 else "rgba(200,30,30,0.2)",
                line=dict(color="rgba(255,255,255,0)"),
                hoverinfo="skip",
                showlegend=False
            )
        )

        # 均值曲线
        fig.add_trace(
            go.Scatter(
                x=steps,
                y=mean_curve,
                mode="lines",
                name=("Baseline(α=0.5)" if alpha == 0.5 else f"Ours(α={ours_alpha})"),
                line=dict(width=2, color="green" if alpha == 0.5 else "red")
                
            )
        )


    line_pos = 0.8 * x_range[1]
    # === 添加 40k 虚线 ===
    fig.add_vline(
        x=line_pos,
        line=dict(color="black", dash="dash", width=2),
        annotation=dict(text="", showarrow=False)  # 强制清空文字
    )
    
    # 在 x 轴下方添加额外ticklabel
    fig.add_annotation(
        x=otick,
        y=-0.01,
        yref="paper",    # y=0 表示 x 轴下边缘
        text=otick_label,
        showarrow=False,
        font=dict(size=32, color="black"),
        xanchor="center",
        yanchor="top"
    )

    # 在虚线附近加文字
    fig.add_annotation(
        x=int(line_pos/2), yref="paper", y=0,  # 左边
        text="Stage 1",
        showarrow=False,
        font=dict(size=28, color="black")
    )
    fig.add_annotation(
        x=int((line_pos+x_range[1])/2), yref="paper", y=0,  # 右边
        text="Stage 2",
        showarrow=False,
        font=dict(size=28, color="black")
    )

    legend_dict = dict(
            x=0.01, y=0.99,
            bgcolor="rgba(255,255,255,0.5)",  # 半透明白色背景
            xanchor="left", yanchor="top",
            font=dict(size=32, color="black")   # 调大 legend 字体
        )
    # if metric == "spearman_corr_test":
    #     legend_dict = dict(
    #         x=0.3, y=0.3,
            # bgcolor="rgba(255,255,255,0.5)",  # 半透明白色背景
    #         xanchor="left", yanchor="top",
    #         font=dict(size=28, color="black")   # 调大 legend 字体
    #     )
    # === 布局设置 ===
    fig.update_layout(
        width=width,
        height=height,
        title=dict(
            text=f"<b>{title}</b>",
            font=dict(size=40, color="black"),
            x=0.5,
            xanchor='center',
            y=0.99,
            yanchor='top'
        ),
        xaxis=dict(
            title="Steps",
            title_font=dict(size=40, color="black"),
            tickfont=dict(size=32, color="black"),
            range=x_range,
            tickmode="array",
            tickvals=ticks,
            ticktext=tick_labels,
            showline=showline,
            linecolor="black",
            linewidth=2,
            mirror=True
        ),
        yaxis=dict(
            title=ylabel,
            title_font=dict(size=40, color="black"),
            tickfont=dict(size=32, color="black"),
            # type="log",
            # range=[0.1, 3]  
            showline=showline,
            linecolor="black",
            linewidth=2,
            mirror=True
        ),
        legend=legend_dict,
        template="plotly_white",
        margin=dict(t=42, b=0, l=30, r=28)
    )

    # === 保存 ===
    os.makedirs('save', exist_ok=True)
    if save_html:
        fig.write_html(os.path.join('save', f"{file_name}.html"), include_mathjax="cdn")
    pio.write_image(fig, os.path.join('save', f"{file_name}.pdf"), width=width, height=height, scale=2)
    if show_pic:
        fig.show()


def calculate_objective(data:dict):
    for _, record in data.items():
        if "fl" in record.keys() and record["fl"]:
            if obj := record.get("objective", None):
                if 'db' in obj:
                    record["Objective"] = "FL-DB"
                elif 'subtb' in obj:
                    record["Objective"] = "FL-SubTB(λ)"
                elif 'tb' in obj:
                    record["Objective"] = "FL-TB"
            elif obj := record.get("method", None):
                if 'db' in obj:
                    record["Objective"] = "FL-DB"
                elif 'subtb' in obj:
                    record["Objective"] = "FL-SubTB(λ)"
                elif 'tb' in obj:
                    record["Objective"] = "FL-TB"
        else:
            if obj := record.get("objective", None):
                if 'db' in obj:
                    record["Objective"] = "DB"
                elif 'subtb' in obj:
                    record["Objective"] = "SubTB(λ)"
                elif 'tb' in obj:
                    record["Objective"] = "TB"
            elif obj := record.get("method", None):
                if 'db' in obj:
                    record["Objective"] = "DB"
                elif 'subtb' in obj:
                    record["Objective"] = "SubTB(λ)"
                elif 'tb' in obj:
                    record["Objective"] = "TB"
    return data

with open("set_run_summary.json", "r") as f:
    data = calculate_objective(json.load(f))