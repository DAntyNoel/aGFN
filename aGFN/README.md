# aGFN（结果汇总 / 绘图脚本）

本目录包含一些用于从导出的 `*.csv`（如 `set.csv` / `mols.csv` / `bit.csv`）里汇总指标、生成图表/表格的脚本与产物文件。

## 关键入口

- `process.py`：主要统计与绘图脚本。
- `parse.py`：把 `<exp>.csv` 按条件过滤后导出成 `<exp>.json`。
- `download_wandb.py`：从 wandb 拉取 runs 并保存为 `mols.csv`（需要 `wandb` 登录）。

## 复现 `alpha_compare_large.pdf`

`alpha_compare_large.pdf` / `alpha_compare_large.png` 由 `process.py` 的 `plot_alpha_compare_large(...)` 生成，并在 `process.py` 的 `__main__` 末尾直接调用：

```bash
python process.py
```

默认读取 `set.csv`，并筛选 `size == "large"` 且 `step == 9999` 的记录后作图，最终通过 `plotly.io.write_image` 导出 PDF/PNG。

## 依赖（导出 PDF/PNG）

- Python 3
- `pandas`, `numpy`
- `plotly` + `kaleido`（`write_image` 需要）

示例安装：

```bash
pip install pandas numpy plotly kaleido
```

