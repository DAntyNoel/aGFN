# Results 生成说明（5-appd / 6-appd）

本文档说明我生成的 `results` 目录内容、处理逻辑以及验证情况，便于合作者复现与审核。

## 生成位置与文件

- `5-appd/results/`
  - `data.csv`：汇总后的均值/标准差表（按 method/size/alpha 分组）
  - `generate_table.py`：生成 LaTeX 表格脚本
  - `new_table.tex`：表格输出
- `6-appd/results/`
  - `data.csv`：汇总后的均值/标准差表（按 method/k/alpha 分组）
  - `generate_table.py`：生成 LaTeX 表格脚本
  - `new_table.tex`：表格输出

## 关键数据源

- `5-appd/all_wandb_histories_merged.jsonl`（实际是 JSON 数组）
- `6-appd/all_wandb_histories_merged.jsonl`

## 处理逻辑（5-appd）

### 1) 去重与续存拼接
- 先按 `(project, run_name)` 分组。
- 对 **Refactored-Alpha-GFN-Set-New-icml-fl0**：
  - 若同名 run 中存在 `step` 非 0 起始的记录，则视为续存。
  - 按 `step` 进行合并（同 step 以后段覆盖前段），得到完整 history。
- 其他项目：从同名 run 的多个片段中选择“**最终 step 最大**”的那条（若并列则选更长的 history / 更晚的时间戳）。

### 2) 过滤条件（与 1-set 的思路一致）
- 所有项目均要求：`step == 9999`, `training_mode == online`, `use_alpha_scheduler == True`, `use_grad_clip == False`, `reward_temp == 1`
- **Refactored-Alpha-GFN-Set-New-icml-fl0** 额外要求：`fl == False`
- **Refactored-Alpha-GFN-Set-New-icml**：**不再用 `fl` 字段过滤**
  - 原因：该项目中 `fl_subtb_gfn` 的 run 在日志里 `fl=True`，若强制 `fl=False` 会导致 **FL-SubTB(large)** 缺失。

### 3) 指标提取与聚合
- 从每条 run 的 history **最后一次出现**的指标值中提取：
  - `modes`, `mean_top_1000_R`, `spearman_corr_test`, `loss`, `forward_policy_entropy_eval`
- 按 `(method, size, alpha)` 计算均值和标准差，生成 `data.csv`。

### 4) 表格输出
- `generate_table.py` 读取 `data.csv`，对每个 size 选择：
  - Baseline：`alpha=0.5`
  - Ours：在 `alpha != 0.5` 中 **modes_mean 最大**
- 表格包含指标：`Modes`, `Top-1000 Reward`, `Spearman Corr`

## 处理逻辑（6-appd）

### 1) 去重
- 按 `(project, run_name)` 分组，选择 **最后 step 最大** 的那条 history。
- 要求 `step >= 49999`（确保完整训练）。

### 2) 指标提取与聚合
- 从 history 的最后一次出现中提取：
  - `modes`, `spearman_corr_test`, `mean_top_1000_R`
- 按 `(method, k, alpha)` 聚合均值/标准差。

### 3) 表格输出
- Baseline：`alpha=0.5`
- Ours：在 `alpha != 0.5` 中 **modes_mean 最大**
- 表格指标：`Modes`, `Spearman`, `MeanTop1kR`

## 验证与检查

- `5-appd/results/data.csv`
  - 行数：**72**
  - methods：`db_gfn, fl_db_gfn, subtb_gfn, fl_subtb_gfn, tb_gfn`
  - sizes：`small/medium/large`
  - `db/fl_db/tb` 各 size **2 个 alpha (0.5, 0.9)**
  - `subtb/fl_subtb` 各 size **9 个 alpha (0.1~0.9)**
  - 无 NaN
- `6-appd/results/data.csv`
  - 行数：**135**
  - methods：`db, subtb, tb`
  - k：`2/4/6/8/10`，每个 method×k 有 **9 个 alpha (0.1~0.9)**
  - 无 NaN
- `new_table.tex` 行尾 `\\` 已修正，LaTeX 可直接编译。

## 复现表格

- 5-appd：
  - `python 5-appd/results/generate_table.py > 5-appd/results/new_table.tex`
- 6-appd：
  - `cd 6-appd/results && python generate_table.py > new_table.tex`

## 备注 / 可选调整

- 如果你希望 **严格复刻 1-set 的过滤（Refactored-Alpha-GFN-Set-New-icml 强制 fl=False）**，
  那么 `FL-SubTB(large)` 将会缺失；目前我为了补全该列，保留了 `fl=True` 的 fl_subtb_gfn。
- 当前表格不展示 `loss` 与 `forward_policy_entropy_eval`，但它们已保留在 `data.csv`。

