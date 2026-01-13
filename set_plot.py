import json
import pandas as pd
import matplotlib.pyplot as plt

# 读取保存的结果
with open("run_summary.json", "r") as f:
    ttl = json.load(f)

# 转换为 DataFrame
records = []
for run_name, summary in ttl.items():
    steps = summary["step"]
    modes = summary["modes"]
    if summary["size"] != 'medium':
        continue
    if modes == 0:
        modes = [0] * len(steps)
    alpha_init = summary["alpha_init"]  # 每个 run 固定值
    for s, m in zip(steps, modes):
        records.append({"run": run_name, "alpha_init": alpha_init, "step": s, "modes": m})

df = pd.DataFrame(records)

# 分组计算均值和标准差
grouped = df.groupby(["alpha_init", "step"])["modes"].agg(["mean", "std"]).reset_index()

# 绘图
plt.figure(figsize=(8,6))
for alpha_val, subdf in grouped.groupby("alpha_init"):
    plt.plot(subdf["step"], subdf["mean"], label=f"alpha_init={alpha_val}")
    plt.fill_between(subdf["step"], 
                     subdf["mean"] - subdf["std"], 
                     subdf["mean"] + subdf["std"], 
                     alpha=0.2)

plt.xlabel("Step")
plt.ylabel("Modes")
plt.title("Modes vs Step grouped by alpha_init")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
