#!/usr/bin/env python3
"""Build visualization artifacts for set-large TB runs with sampling_seed in {0,1,2}.

This script follows the same filtering logic used in `1-set/parse.py`:
- step == 9999
- training_mode == "online"
- use_alpha_scheduler == True
- use_grad_clip == False
- reward_temp == 1
- project-specific `fl` handling for each source file

Outputs under this folder:
- set_large_tb_seed012_raw.csv
- set_large_tb_seed012_summary.csv
- set_large_tb_seed012_plot.png (if matplotlib is available)
"""

from __future__ import annotations

import json
import csv
import math
from pathlib import Path
from typing import Any, Dict, List
from statistics import mean, stdev

ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parent.parent
SET_ROOT = REPO_ROOT / "1-set"

SOURCE_FILES = [
    "rebuttal_set_temp_old.json",
    "refactored_alpha_gfn_set_new_icml.json",
    "refactored_alpha_gfn_set_new_icml_fl0.json",
    "rebuttal_set_fl.json",
]

TARGET_SIZE = "large"
TARGET_METHOD = "tb_gfn"
TARGET_SEEDS = {0, 1, 2}

RAW_OUT = ROOT / "set_large_tb_seed012_raw.csv"
SUMMARY_OUT = ROOT / "set_large_tb_seed012_summary.csv"
PLOT_OUT = ROOT / "set_large_tb_seed012_plot.png"


def is_base_valid(summary: Dict[str, Any]) -> bool:
    return (
        summary.get("step") == 9999
        and summary.get("training_mode") == "online"
        and summary.get("use_alpha_scheduler") is True
        and summary.get("use_grad_clip") is False
        and summary.get("reward_temp") == 1
    )


def pass_project_filter(source_name: str, summary: Dict[str, Any]) -> bool:
    # Match the project-specific behavior in 1-set/parse.py.
    if source_name == "refactored_alpha_gfn_set_new_icml.json":
        return summary.get("fl") is True
    if source_name == "refactored_alpha_gfn_set_new_icml_fl0.json":
        return summary.get("fl") is False
    return True


def load_rows() -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for source_name in SOURCE_FILES:
        source_path = SET_ROOT / source_name
        if not source_path.exists():
            continue

        with source_path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        for run_name, summary in data.items():
            if not isinstance(summary, dict):
                continue
            if not is_base_valid(summary):
                continue
            if not pass_project_filter(source_name, summary):
                continue

            if summary.get("size") != TARGET_SIZE:
                continue
            if summary.get("method") != TARGET_METHOD:
                continue
            if summary.get("sampling_seed") not in TARGET_SEEDS:
                continue

            rows.append(
                {
                    "source_file": source_name,
                    "run_name": run_name,
                    "sampling_seed": summary.get("sampling_seed"),
                    "alpha_init": summary.get("alpha_init"),
                    "method": summary.get("method"),
                    "size": summary.get("size"),
                    "modes": summary.get("modes"),
                    "mean_top_1000_R": summary.get("mean_top_1000_R"),
                    "mean_top_1000_similarity": summary.get("mean_top_1000_similarity"),
                    "spearman_corr_test": summary.get("spearman_corr_test"),
                }
            )

    return rows


def to_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def write_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: List[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def safe_mean(values: List[float]) -> float | None:
    return mean(values) if values else None


def safe_std(values: List[float]) -> float | None:
    if len(values) <= 1:
        return 0.0 if len(values) == 1 else None
    return stdev(values)


def main() -> None:
    rows = load_rows()
    if not rows:
        raise SystemExit("No rows found for set large + tb + sampling_seed in {0,1,2}.")

    rows = sorted(
        rows,
        key=lambda r: (
            to_float(r.get("alpha_init")) if to_float(r.get("alpha_init")) is not None else math.inf,
            int(r.get("sampling_seed")) if r.get("sampling_seed") is not None else 999,
            str(r.get("source_file")),
            str(r.get("run_name")),
        ),
    )

    raw_fields = [
        "source_file",
        "run_name",
        "sampling_seed",
        "alpha_init",
        "method",
        "size",
        "modes",
        "mean_top_1000_R",
        "mean_top_1000_similarity",
        "spearman_corr_test",
    ]
    write_csv(RAW_OUT, rows, raw_fields)

    grouped: Dict[float, List[Dict[str, Any]]] = {}
    for row in rows:
        alpha = to_float(row.get("alpha_init"))
        if alpha is None:
            continue
        grouped.setdefault(alpha, []).append(row)

    summary_rows: List[Dict[str, Any]] = []
    for alpha in sorted(grouped.keys()):
        grp = grouped[alpha]

        modes_vals = [v for v in (to_float(x.get("modes")) for x in grp) if v is not None]
        reward_vals = [v for v in (to_float(x.get("mean_top_1000_R")) for x in grp) if v is not None]
        sim_vals = [v for v in (to_float(x.get("mean_top_1000_similarity")) for x in grp) if v is not None]
        spear_vals = [v for v in (to_float(x.get("spearman_corr_test")) for x in grp) if v is not None]

        summary_rows.append(
            {
                "alpha_init": alpha,
                "n_runs": len(grp),
                "modes_mean": safe_mean(modes_vals),
                "modes_std": safe_std(modes_vals),
                "mean_top_1000_R_mean": safe_mean(reward_vals),
                "mean_top_1000_R_std": safe_std(reward_vals),
                "mean_top_1000_similarity_mean": safe_mean(sim_vals),
                "mean_top_1000_similarity_std": safe_std(sim_vals),
                "spearman_corr_test_mean": safe_mean(spear_vals),
                "spearman_corr_test_std": safe_std(spear_vals),
            }
        )

    summary_fields = [
        "alpha_init",
        "n_runs",
        "modes_mean",
        "modes_std",
        "mean_top_1000_R_mean",
        "mean_top_1000_R_std",
        "mean_top_1000_similarity_mean",
        "mean_top_1000_similarity_std",
        "spearman_corr_test_mean",
        "spearman_corr_test_std",
    ]
    write_csv(SUMMARY_OUT, summary_rows, summary_fields)

    try:
        import matplotlib.pyplot as plt  # pylint: disable=import-outside-toplevel

        plt.style.use("seaborn-v0_8-whitegrid")
        fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
        metrics = [
            ("modes_mean", "modes_std", "Modes"),
            ("mean_top_1000_R_mean", "mean_top_1000_R_std", "Top-1000 Reward"),
            (
                "mean_top_1000_similarity_mean",
                "mean_top_1000_similarity_std",
                "Top-1000 Similarity",
            ),
            ("spearman_corr_test_mean", "spearman_corr_test_std", "Spearman Corr."),
        ]

        for ax, (m_col, s_col, title) in zip(axes.ravel(), metrics):
            x = [float(r["alpha_init"]) for r in summary_rows]
            y = [to_float(r[m_col]) for r in summary_rows]
            yerr = [to_float(r[s_col]) or 0.0 for r in summary_rows]
            ax.errorbar(x, y, yerr=yerr, marker="o", capsize=4, linewidth=2)
            ax.set_title(title)
            ax.set_xlabel("alpha_init")
            ax.set_ylabel(title)

        fig.suptitle("Set Large | TB | sampling_seed = 0/1/2", fontsize=14)
        fig.savefig(PLOT_OUT, dpi=200)
        plt.close(fig)
        print(f"Saved plot: {PLOT_OUT}")
    except Exception as exc:  # noqa: BLE001
        print(f"Plot skipped: {exc}")

    print(f"Saved raw rows: {RAW_OUT}")
    print(f"Saved summary: {SUMMARY_OUT}")
    print(f"Total matched runs: {len(rows)}")


if __name__ == "__main__":
    main()
