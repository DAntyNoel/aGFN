#!/usr/bin/env python3
"""Deep-check PBP metrics and build tables/plots for target concepts.

Input:
- pbp_alpha_diversity_0327.csv

Outputs:
- pbp_metric_inventory.csv
- pbp_metric_mapping.csv
- pbp_target_candidates_overall.csv
- pbp_target_candidates_by_alpha.csv
- pbp_target_selected_by_alpha.csv
- pbp_target_candidates_plot.png
- pbp_target_selected_plot.png
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

ROOT = Path(__file__).resolve().parent
IN_CSV = ROOT / "pbp_alpha_diversity_0327.csv"

OUT_INVENTORY = ROOT / "pbp_metric_inventory.csv"
OUT_MAPPING = ROOT / "pbp_metric_mapping.csv"
OUT_CANDIDATES_OVERALL = ROOT / "pbp_target_candidates_overall.csv"
OUT_CANDIDATES_ALPHA = ROOT / "pbp_target_candidates_by_alpha.csv"
OUT_SELECTED_ALPHA = ROOT / "pbp_target_selected_by_alpha.csv"
OUT_PLOT_CANDIDATES = ROOT / "pbp_target_candidates_plot.png"
OUT_PLOT_SELECTED = ROOT / "pbp_target_selected_plot.png"

TARGETS = ["modes", "spearman correlation", "diversity", "top rewards"]

CANDIDATE_RULES = {
    "modes": [
        "paper_num_modes",
        "all_num_modes",
        "onpolicy_num_modes",
        "paper_onpolicy_num_modes",
        "paper_num_modes_frac",
    ],
    "spearman correlation": [],
    "diversity": [
        "top1000_tanimoto_diversity",
        "top100_tanimoto_diversity",
        "onpolicy_tanimoto_diversity",
        "onpolicy_mode_tanimoto_diversity",
        "entropy",
    ],
    "top rewards": [
        "sampled_reward_avg",
        "reward_loss",
    ],
}


def parse_json_cell(value: Any) -> Dict[str, Any]:
    if isinstance(value, dict):
        return value
    if pd.isna(value):
        return {}
    return json.loads(value)


def mean_pm_std(series: pd.Series) -> str:
    series = pd.to_numeric(series, errors="coerce").dropna()
    if len(series) == 0:
        return "N/A"
    mean = series.mean()
    std = series.std(ddof=1) if len(series) > 1 else 0.0
    return f"{mean:.6g}+-{std:.6g}"


def build_inventory(summary_dicts: List[Dict[str, Any]]) -> pd.DataFrame:
    key_stats: Dict[str, Dict[str, float]] = {}
    total = len(summary_dicts)

    for summary in summary_dicts:
        for k, v in summary.items():
            stat = key_stats.setdefault(k, {"present_count": 0, "numeric_count": 0})
            stat["present_count"] += 1
            numeric = pd.to_numeric(pd.Series([v]), errors="coerce").iloc[0]
            if pd.notna(numeric):
                stat["numeric_count"] += 1

    rows = []
    for k in sorted(key_stats):
        stat = key_stats[k]
        rows.append(
            {
                "metric_key": k,
                "present_count": int(stat["present_count"]),
                "present_ratio": stat["present_count"] / total if total else 0,
                "numeric_count": int(stat["numeric_count"]),
                "numeric_ratio": stat["numeric_count"] / total if total else 0,
            }
        )

    return pd.DataFrame(rows)


def choose_metric_for_modes(df: pd.DataFrame, available: List[str]) -> tuple[str, str, str]:
    # paper_num_modes is preferred semantically, but if constant zero then it is not informative.
    for metric in ["paper_num_modes", "all_num_modes", "onpolicy_num_modes", "paper_onpolicy_num_modes", "paper_num_modes_frac"]:
        if metric not in available:
            continue
        vals = pd.to_numeric(df[metric], errors="coerce").dropna()
        if len(vals) == 0:
            continue
        if vals.nunique(dropna=True) <= 1:
            if metric == "paper_num_modes":
                continue
            return metric, "proxy", f"{metric} is available but constant-check may be weak (unique={vals.nunique(dropna=True)})."
        if metric == "paper_num_modes":
            return metric, "available", "paper_num_modes varies and is directly aligned with paper-level mode count."
        return metric, "proxy", "paper_num_modes is constant; switched to the first non-constant mode-count proxy."
    return "N/A", "missing", "No usable mode-count metric found."


def choose_metric_for_target(target: str, df: pd.DataFrame, available: List[str]) -> tuple[str, str, str]:
    if target == "modes":
        return choose_metric_for_modes(df, available)

    if target == "spearman correlation":
        corr_like = [k for k in available if ("spearman" in k.lower() or "corr" in k.lower())]
        if corr_like:
            return corr_like[0], "available", "Direct spearman/correlation metric found."
        return "N/A", "missing", "No metric containing spearman/corr in this project (checked summary keys)."

    if target == "diversity":
        for metric in CANDIDATE_RULES[target]:
            if metric in available:
                return metric, "available", "Selected by priority: top-k tanimoto diversity is closest to global diversity quality."
        return "N/A", "missing", "No diversity-like metric found."

    if target == "top rewards":
        for metric in CANDIDATE_RULES[target]:
            if metric in available:
                status = "proxy" if metric == "sampled_reward_avg" else "proxy"
                return metric, status, "No explicit top-k reward metric; using reward-level proxy by priority."
        return "N/A", "missing", "No reward-like metric found."

    return "N/A", "missing", "Unsupported target."


def compute_candidate_tables(df: pd.DataFrame, targets_to_candidates: Dict[str, List[str]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    overall_rows: List[Dict[str, Any]] = []
    alpha_rows: List[Dict[str, Any]] = []

    grouped_alpha = df.groupby("alpha_init")

    for target, metrics in targets_to_candidates.items():
        if not metrics:
            overall_rows.append(
                {
                    "target": target,
                    "metric": "N/A",
                    "n_runs": len(df),
                    "mean": pd.NA,
                    "std": pd.NA,
                    "mean+-std": "N/A",
                    "available": False,
                }
            )
            for alpha, g in grouped_alpha:
                alpha_rows.append(
                    {
                        "target": target,
                        "metric": "N/A",
                        "alpha_init": alpha,
                        "n_runs": len(g),
                        "mean": pd.NA,
                        "std": pd.NA,
                        "mean+-std": "N/A",
                        "available": False,
                    }
                )
            continue

        for metric in metrics:
            vals = pd.to_numeric(df[metric], errors="coerce")
            valid = vals.dropna()
            overall_rows.append(
                {
                    "target": target,
                    "metric": metric,
                    "n_runs": int(valid.shape[0]),
                    "mean": valid.mean() if len(valid) else pd.NA,
                    "std": valid.std(ddof=1) if len(valid) > 1 else (0.0 if len(valid) == 1 else pd.NA),
                    "mean+-std": mean_pm_std(vals),
                    "available": len(valid) > 0,
                }
            )

            for alpha, g in grouped_alpha:
                gvals = pd.to_numeric(g[metric], errors="coerce")
                gvalid = gvals.dropna()
                alpha_rows.append(
                    {
                        "target": target,
                        "metric": metric,
                        "alpha_init": alpha,
                        "n_runs": int(g.shape[0]),
                        "mean": gvalid.mean() if len(gvalid) else pd.NA,
                        "std": gvalid.std(ddof=1) if len(gvalid) > 1 else (0.0 if len(gvalid) == 1 else pd.NA),
                        "mean+-std": mean_pm_std(gvals),
                        "available": len(gvalid) > 0,
                    }
                )

    overall_df = pd.DataFrame(overall_rows)
    alpha_df = pd.DataFrame(alpha_rows)
    return overall_df, alpha_df


def build_selected_alpha(alpha_df: pd.DataFrame, mapping_df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for _, map_row in mapping_df.iterrows():
        target = map_row["target_concept"]
        metric = map_row["selected_metric"]
        status = map_row["status"]

        sub = alpha_df[(alpha_df["target"] == target) & (alpha_df["metric"] == metric)]
        if sub.empty and metric == "N/A":
            # keep explicit missing rows per alpha for plotting/reporting
            template = alpha_df[alpha_df["target"] == target][["alpha_init", "n_runs"]].drop_duplicates()
            for _, trow in template.iterrows():
                rows.append(
                    {
                        "target": target,
                        "selected_metric": metric,
                        "status": status,
                        "alpha_init": trow["alpha_init"],
                        "n_runs": trow["n_runs"],
                        "mean": pd.NA,
                        "std": pd.NA,
                        "mean+-std": "N/A",
                    }
                )
            continue

        for _, r in sub.iterrows():
            rows.append(
                {
                    "target": target,
                    "selected_metric": metric,
                    "status": status,
                    "alpha_init": r["alpha_init"],
                    "n_runs": r["n_runs"],
                    "mean": r["mean"],
                    "std": r["std"],
                    "mean+-std": r["mean+-std"],
                }
            )

    return pd.DataFrame(rows)


def make_plots(alpha_df: pd.DataFrame, mapping_df: pd.DataFrame) -> None:
    try:
        import matplotlib.pyplot as plt  # pylint: disable=import-outside-toplevel
    except Exception as exc:  # noqa: BLE001
        print(f"Plot skipped (matplotlib unavailable): {exc}")
        return

    plt.style.use("seaborn-v0_8-whitegrid")

    # Figure 1: all candidates for each target.
    fig1, axes1 = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
    for ax, target in zip(axes1.ravel(), TARGETS):
        sub = alpha_df[(alpha_df["target"] == target) & (alpha_df["available"] == True)]  # noqa: E712
        if sub.empty:
            ax.text(0.5, 0.5, "No available metrics", ha="center", va="center")
            ax.set_title(target)
            ax.set_xlabel("alpha_init")
            ax.set_ylabel("value")
            continue

        for metric in sorted(sub["metric"].unique()):
            msub = sub[sub["metric"] == metric].sort_values("alpha_init")
            x = msub["alpha_init"].tolist()
            y = pd.to_numeric(msub["mean"], errors="coerce").tolist()
            yerr = pd.to_numeric(msub["std"], errors="coerce").fillna(0).tolist()
            ax.errorbar(x, y, yerr=yerr, marker="o", capsize=3, linewidth=1.8, label=metric)

        ax.set_title(target)
        ax.set_xlabel("alpha_init")
        ax.set_ylabel("value")
        ax.legend(fontsize=8)

    fig1.suptitle("PBP target candidates (mean +- std by alpha)", fontsize=14)
    fig1.savefig(OUT_PLOT_CANDIDATES, dpi=220)
    plt.close(fig1)

    # Figure 2: selected metric per target.
    fig2, axes2 = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for ax, target in zip(axes2.ravel(), TARGETS):
        row = mapping_df[mapping_df["target_concept"] == target].iloc[0]
        metric = row["selected_metric"]

        if metric == "N/A":
            ax.text(0.5, 0.5, "No direct metric", ha="center", va="center")
            ax.set_title(f"{target} (missing)")
            ax.set_xlabel("alpha_init")
            ax.set_ylabel("value")
            continue

        sub = alpha_df[(alpha_df["target"] == target) & (alpha_df["metric"] == metric)].sort_values("alpha_init")
        x = sub["alpha_init"].tolist()
        y = pd.to_numeric(sub["mean"], errors="coerce").tolist()
        yerr = pd.to_numeric(sub["std"], errors="coerce").fillna(0).tolist()

        ax.errorbar(x, y, yerr=yerr, marker="o", capsize=4, linewidth=2.2)
        ax.set_title(f"{target}: {metric}")
        ax.set_xlabel("alpha_init")
        ax.set_ylabel(metric)

    fig2.suptitle("PBP selected metrics (mean +- std by alpha)", fontsize=14)
    fig2.savefig(OUT_PLOT_SELECTED, dpi=220)
    plt.close(fig2)

    print(f"saved plot: {OUT_PLOT_CANDIDATES}")
    print(f"saved plot: {OUT_PLOT_SELECTED}")


def main() -> None:
    if not IN_CSV.exists():
        raise SystemExit(f"Input not found: {IN_CSV}. Run download_wandb.py first.")

    df = pd.read_csv(IN_CSV)

    summary_dicts = [parse_json_cell(x) for x in df["summary"].tolist()]
    config_dicts = [parse_json_cell(x) for x in df["config"].tolist()]

    # Flatten key metrics to a dataframe for analysis.
    data_rows: List[Dict[str, Any]] = []
    for base, summary, config in zip(df.to_dict(orient="records"), summary_dicts, config_dicts):
        row = {
            "name": base.get("name"),
            "alpha_init": config.get("alpha"),
        }
        row.update(summary)
        data_rows.append(row)

    work_df = pd.DataFrame(data_rows)
    work_df["alpha_init"] = pd.to_numeric(work_df["alpha_init"], errors="coerce")

    inventory_df = build_inventory(summary_dicts)
    inventory_df.to_csv(OUT_INVENTORY, index=False)

    available_metrics = set(inventory_df["metric_key"].tolist())
    targets_to_candidates: Dict[str, List[str]] = {}
    mapping_rows: List[Dict[str, Any]] = []

    for target in TARGETS:
        raw_candidates = CANDIDATE_RULES[target]
        present_candidates = [m for m in raw_candidates if m in available_metrics]
        targets_to_candidates[target] = present_candidates

        selected_metric, status, notes = choose_metric_for_target(target, work_df, list(available_metrics))

        mapping_rows.append(
            {
                "target_concept": target,
                "selected_metric": selected_metric,
                "status": status,
                "candidate_metrics": ";".join(present_candidates) if present_candidates else "N/A",
                "notes": notes,
            }
        )

    mapping_df = pd.DataFrame(mapping_rows)
    mapping_df.to_csv(OUT_MAPPING, index=False)

    overall_df, alpha_df = compute_candidate_tables(work_df, targets_to_candidates)
    overall_df.to_csv(OUT_CANDIDATES_OVERALL, index=False)
    alpha_df.sort_values(["target", "metric", "alpha_init"]).to_csv(OUT_CANDIDATES_ALPHA, index=False)

    selected_alpha_df = build_selected_alpha(alpha_df, mapping_df)
    selected_alpha_df.sort_values(["target", "alpha_init"]).to_csv(OUT_SELECTED_ALPHA, index=False)

    # Explicit diagnostic for the user's question about paper_num_modes.
    if "paper_num_modes" in work_df.columns:
        pvals = pd.to_numeric(work_df["paper_num_modes"], errors="coerce").dropna()
        print(
            "paper_num_modes diagnostic:",
            {
                "n": int(pvals.shape[0]),
                "mean": float(pvals.mean()) if len(pvals) else None,
                "std": float(pvals.std(ddof=1)) if len(pvals) > 1 else 0.0,
                "unique": int(pvals.nunique()) if len(pvals) else 0,
                "min": float(pvals.min()) if len(pvals) else None,
                "max": float(pvals.max()) if len(pvals) else None,
            },
        )
    if "all_num_modes" in work_df.columns:
        avals = pd.to_numeric(work_df["all_num_modes"], errors="coerce").dropna()
        print(
            "all_num_modes diagnostic:",
            {
                "n": int(avals.shape[0]),
                "mean": float(avals.mean()) if len(avals) else None,
                "std": float(avals.std(ddof=1)) if len(avals) > 1 else 0.0,
                "unique": int(avals.nunique()) if len(avals) else 0,
                "min": float(avals.min()) if len(avals) else None,
                "max": float(avals.max()) if len(avals) else None,
            },
        )

    make_plots(alpha_df, mapping_df)

    print(f"saved inventory: {OUT_INVENTORY}")
    print(f"saved mapping: {OUT_MAPPING}")
    print(f"saved candidates overall: {OUT_CANDIDATES_OVERALL}")
    print(f"saved candidates by alpha: {OUT_CANDIDATES_ALPHA}")
    print(f"saved selected by alpha: {OUT_SELECTED_ALPHA}")
    print(f"total runs: {len(work_df)}")


if __name__ == "__main__":
    main()
