import json
import re
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
import pandas as pd

MERGED_JSON = Path("wandb_histories_merged.json")
OUTPUT_FILTERED_JSON = Path("filtered_histories.json")
OUTPUT_SUMMARY_CSV = Path("filtered_histories_summary.csv")

PROJECT_TO_SOURCE_FILE = {
    "Rebuttal-Set-Temp-Old": "rebuttal_set_temp_old.json",
    "Refactored-Alpha-GFN-Set-New-icml": "refactored_alpha_gfn_set_new_icml.json",
    "Refactored-Alpha-GFN-Set-New-icml-fl0": "refactored_alpha_gfn_set_new_icml_fl0.json",
    "Rebuttal-Set-FL": "rebuttal_set_fl.json",
}

DEFAULT_METRICS = [
    "modes",
    "mean_top_1000_R",
    "mean_top_1000_similarity",
    "spearman_corr_test",
]


def flatten_dict(d: Dict[str, Any], parent_key: str = "", sep: str = "_") -> Dict[str, Any]:
    items: List[tuple[str, Any]] = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def parse_exp_name(exp_name: str) -> dict:
    params = {}
    ss_match = re.search(r"ss\((\d+)\)", exp_name)
    if ss_match:
        params["seed"] = int(ss_match.group(1))

    m_match = re.search(r"_m\(([^)]+)\)", exp_name)
    if m_match:
        params["method"] = m_match.group(1)

    sz_match = re.search(r"_sz\(([^)]+)\)", exp_name)
    if sz_match:
        params["size"] = sz_match.group(1)

    a_match = re.search(r"_a\(([^)]+)\)", exp_name)
    if a_match:
        try:
            params["alpha_init_from_name"] = float(a_match.group(1))
        except ValueError:
            params["alpha_init_from_name"] = np.nan

    return params


def load_merged() -> Dict[str, Dict[str, Any]]:
    if not MERGED_JSON.exists():
        raise FileNotFoundError(f"Merged json not found: {MERGED_JSON}")
    with MERGED_JSON.open("r") as f:
        return json.load(f)


def normalize_scalar(value: Any) -> Any:
    if isinstance(value, (list, tuple, np.ndarray)):
        if len(value) == 0:
            return np.nan
        return value[-1]
    return value


def main():
    merged = load_merged()

    all_rows = []
    for project, runs in merged.items():
        source_file = PROJECT_TO_SOURCE_FILE.get(project, project)
        for exp_name, exp_params in runs.items():
            flat_params = flatten_dict(exp_params)
            flat_params["experiment_name"] = exp_name
            flat_params["source_file"] = source_file
            all_rows.append(flat_params)

    df = pd.DataFrame(all_rows)
    print(f"✓ 加载完成！共 {len(df)} 行")
    print(f"✓ 列数: {len(df.columns)}")
    print(f"\n数据源: {sorted(df['source_file'].unique().tolist())}")
    print(f"\nDataFrame 形状: {df.shape}")

    df_all = df.copy()
    parsed_rows = []
    for _, row in df_all.iterrows():
        name = row.get("experiment_name", "")
        parsed = parse_exp_name(name)
        method = parsed.get("method")

        source_file = row.get("source_file", "")
        if source_file in ["refactored_alpha_gfn_set_new_icml.json"]:
            if method:
                method = f"fl_{method}"

        parsed_rows.append(
            {
                **row,
                "seed": parsed.get("seed"),
                "method": method,
                "size": parsed.get("size"),
                "alpha_init_name": parsed.get("alpha_init_from_name"),
                "alpha": normalize_scalar(row.get("alpha_init", row.get("alpha"))),
            }
        )

    df_all = pd.DataFrame(parsed_rows)

    for col in ["alpha", *DEFAULT_METRICS]:
        if col in df_all.columns:
            df_all[col] = df_all[col].apply(normalize_scalar)
            df_all[col] = pd.to_numeric(df_all[col], errors="coerce")

    before_filter = len(df_all)
    df_all = df_all[
        ~(
            (df_all["source_file"] == "rebuttal_set_fl.json")
            & (df_all["method"] == "subtb_gfn")
        )
    ]
    after_filter = len(df_all)
    if before_filter > after_filter:
        print(
            f"✓ 已过滤掉 {before_filter - after_filter} 条数据（来自 rebuttal_set_fl.json 且 method=subtb_gfn）\n"
        )

    print(f"总实验数: {len(df_all)}")
    print(f"Unique seeds: {sorted(df_all['seed'].dropna().unique())}")
    print(f"Unique sizes: {sorted(df_all['size'].dropna().unique())}")
    print(f"Unique alphas (from alpha_init): {sorted(df_all['alpha'].dropna().unique())}")
    print(f"Unique objectives/methods: {sorted(df_all['method'].dropna().unique())}")
    print("\n数据来源分布:")
    print(df_all.groupby(["source_file", "method", "size"]).size())

    metrics = DEFAULT_METRICS.copy()
    available_metrics = [m for m in metrics if m in df_all.columns]
    missing_metrics = [m for m in metrics if m not in df_all.columns]

    if missing_metrics:
        print(f"⚠️ 缺失指标: {missing_metrics}")
        print(f"✓ 可用指标: {available_metrics}\n")
        metrics = available_metrics

    objectives = sorted(df_all["method"].dropna().unique())
    sizes = sorted(df_all["size"].dropna().unique())

    print(f"\n{'=' * 80}")
    print(f"找到 {len(objectives)} 个 objectives: {objectives}")
    print(f"找到 {len(sizes)} 个 sizes: {sizes}")
    print(f"{'=' * 80}")

    for obj in objectives:
        for size in sizes:
            print(f"\n{'=' * 80}")
            print(f"Objective: {obj}, Size: {size}")
            print(f"{'=' * 80}")

            df_obj = df_all[(df_all["method"] == obj) & (df_all["size"] == size)].copy()
            if len(df_obj) == 0:
                print("无数据，跳过")
                continue

            print(f"  实验数: {len(df_obj)}")
            print(f"  来源文件: {df_obj['source_file'].unique()}")

            if metrics:
                df_averaged = df_obj.groupby("alpha")[metrics].mean().reset_index()
                df_averaged = df_averaged.sort_values("alpha")

                print("\n按 seed 平均后的结果 (基于 alpha_init):")
                print(df_averaged.to_string(index=False))

                print("\n最优 Alpha 分析:")
                print(f"{'-' * 80}")
                for metric in metrics:
                    if len(df_averaged) > 0 and not df_averaged[metric].isna().all():
                        best_idx = df_averaged[metric].idxmax()
                        best_alpha = df_averaged.loc[best_idx, "alpha"]
                        best_value = df_averaged.loc[best_idx, metric]
                        print(
                            f"  {metric:30s} -> 最优 alpha_init: {best_alpha:.4f}, 最优值: {best_value:.6f}"
                        )

    df_filtered = df_all[
        df_all["method"].notna()
        & df_all["size"].notna()
        & df_all["alpha"].notna()
    ].copy()

    df_filtered.to_csv(OUTPUT_SUMMARY_CSV, index=False)

    filtered_histories: Dict[str, Dict[str, Any]] = {}
    for _, row in df_filtered.iterrows():
        source = row.get("source_file", "unknown")
        exp_name = row.get("experiment_name", "")
        if source not in filtered_histories:
            filtered_histories[source] = {}
        filtered_histories[source][exp_name] = row.dropna().to_dict()

    with OUTPUT_FILTERED_JSON.open("w") as f:
        json.dump(filtered_histories, f, indent=2)

    print(f"\n✓ 已保存筛选后的汇总表: {OUTPUT_SUMMARY_CSV}")
    print(f"✓ 已保存筛选后的 history: {OUTPUT_FILTERED_JSON}")


if __name__ == "__main__":
    main()
