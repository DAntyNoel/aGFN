#!/usr/bin/env python3
"""Plot Appd figures following AppdD2.ipynb style.

- Set (from 5-appd/all_wandb_histories_merged_resumed.json if present,
  otherwise 5-appd/all_wandb_histories_merged.jsonl)
- Bit (from 6-appd/all_wandb_histories_merged.jsonl)

Outputs are saved under:
- 5-appd/save
- 6-appd/save

PDF export requires kaleido; if unavailable, HTML is saved.
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path
from collections import defaultdict

import pandas as pd

import numpy as np
import plotly.graph_objects as go
import plotly.io as pio

ROOT = Path(__file__).resolve().parent

SET_JSON = ROOT / "5-appd" / "all_wandb_histories_merged_resumed.json"
SET_JSON_FALLBACK = ROOT / "5-appd" / "all_wandb_histories_merged.jsonl"
BIT_JSON = ROOT / "6-appd" / "all_wandb_histories_merged.jsonl"

SET_SAVE = ROOT / "5-appd" / "save"
BIT_SAVE = ROOT / "6-appd" / "save"
MOLS_JSON = ROOT / "7-mols" / "mols_run_summary.json"
MOLS_SAVE = ROOT / "7-mols" / "save"

ONE_SET_BEST = ROOT / "1-set" / "results" / "data.csv"
TWO_BIT_BEST = ROOT / "2-bit" / "results" / "data.csv"

SAVE_HTML = False
SAVE_PDF = True  # requires kaleido


# ----------------------------- plotting utils -----------------------------

def build_image(
    steps,
    groups,
    ours_alpha,
    title,
    ylabel,
    metric,
    file_name,
    x_range=None,
    ticks=None,
    tick_labels=None,
    otick=None,
    otick_label=None,
    width=800,
    height=600,
    showline=True,
    save_dir=SET_SAVE,
):
    """Build and save figure. Uses nanmean/nanstd for robustness."""
    fig = go.Figure()
    alpha_list = [0.5, ours_alpha]

    # Filter out steps with all-NaN across both baseline and ours
    runs_base = np.array(groups.get(0.5, []), dtype=float)
    runs_ours = np.array(groups.get(ours_alpha, []), dtype=float)
    if runs_base.size == 0 or runs_ours.size == 0:
        return
    valid_mask = np.any(np.isfinite(runs_base), axis=0) | np.any(np.isfinite(runs_ours), axis=0)
    if not np.any(valid_mask):
        return

    steps = np.array(steps, dtype=float)[valid_mask]

    for alpha in alpha_list:
        runs = np.array(groups[alpha], dtype=float)[:, valid_mask]  # shape: (n_runs, n_steps)
        mean_curve = np.nanmean(runs, axis=0)
        std_curve = np.nanstd(runs, axis=0)

        fig.add_trace(
            go.Scatter(
                x=steps.tolist() + steps[::-1].tolist(),
                y=(mean_curve + std_curve).tolist() + (mean_curve - std_curve)[::-1].tolist(),
                fill="toself",
                fillcolor="rgba(0,100,80,0.2)" if alpha == 0.5 else "rgba(200,30,30,0.2)",
                line=dict(color="rgba(255,255,255,0)"),
                hoverinfo="skip",
                showlegend=False,
            )
        )

        fig.add_trace(
            go.Scatter(
                x=steps,
                y=mean_curve,
                mode="lines",
                name=("Baseline(α=0.5)" if alpha == 0.5 else f"Ours(α={ours_alpha})"),
                line=dict(width=2, color="green" if alpha == 0.5 else "red"),
            )
        )

    if x_range is None:
        x_range = [float(np.min(steps)), float(np.max(steps))]
    if ticks is None:
        ticks = []
    if tick_labels is None:
        tick_labels = []

    line_pos = 0.8 * x_range[1]
    fig.add_vline(
        x=line_pos,
        line=dict(color="black", dash="dash", width=2),
        annotation=dict(text="", showarrow=False),
    )

    if otick is not None and otick_label is not None:
        fig.add_annotation(
            x=otick,
            y=-0.01,
            yref="paper",
            text=otick_label,
            showarrow=False,
            font=dict(size=32, color="black"),
            xanchor="center",
            yanchor="top",
        )

    fig.add_annotation(
        x=int(line_pos / 2),
        yref="paper",
        y=0,
        text="Stage 1",
        showarrow=False,
        font=dict(size=28, color="black"),
    )
    fig.add_annotation(
        x=int((line_pos + x_range[1]) / 2),
        yref="paper",
        y=0,
        text="Stage 2",
        showarrow=False,
        font=dict(size=28, color="black"),
    )

    legend_dict = dict(
        x=0.01,
        y=0.99,
        bgcolor="rgba(255,255,255,0.5)",
        xanchor="left",
        yanchor="top",
        font=dict(size=32, color="black"),
    )

    fig.update_layout(
        width=width,
        height=height,
        title=dict(
            text=f"<b>{title}</b>",
            font=dict(size=40, color="black"),
            x=0.5,
            xanchor="center",
            y=0.99,
            yanchor="top",
        ),
        xaxis=dict(
            title="Steps",
            title_font=dict(size=40, color="black"),
            tickfont=dict(size=32, color="black"),
            range=x_range,
            tickmode="array" if ticks else "auto",
            tickvals=ticks,
            ticktext=tick_labels,
            showline=showline,
            linecolor="black",
            linewidth=2,
            mirror=True,
        ),
        yaxis=dict(
            title=ylabel,
            title_font=dict(size=40, color="black"),
            tickfont=dict(size=32, color="black"),
            showline=showline,
            linecolor="black",
            linewidth=2,
            mirror=True,
        ),
        legend=legend_dict,
        template="plotly_white",
        margin=dict(t=42, b=0, l=30, r=28),
    )

    os.makedirs(save_dir, exist_ok=True)

    wrote_html = False
    if SAVE_HTML:
        fig.write_html(str(Path(save_dir) / f"{file_name}.html"), include_mathjax="cdn")
        wrote_html = True

    if SAVE_PDF:
        try:
            pio.write_image(fig, str(Path(save_dir) / f"{file_name}.pdf"), width=width, height=height, scale=2)
        except Exception as exc:  # noqa: BLE001
            print(f"[WARN] PDF export failed ({file_name}): {exc}")
            if not wrote_html:
                fig.write_html(str(Path(save_dir) / f"{file_name}.html"), include_mathjax="cdn")


# ----------------------------- helpers -----------------------------

def calculate_objective(record: dict) -> str | None:
    if record.get("fl"):
        obj = record.get("objective") or record.get("method")
        if not obj:
            return None
        if "db" in obj:
            return "FL-DB"
        if "subtb" in obj:
            return "FL-SubTB(λ)"
        if "tb" in obj:
            return "FL-TB"
    else:
        obj = record.get("objective") or record.get("method")
        if not obj:
            return None
        if "db" in obj:
            return "DB"
        if "subtb" in obj:
            return "SubTB(λ)"
        if "tb" in obj:
            return "TB"
    return None


def parse_run_name(name: str) -> dict:
    out = {}
    m = re.search(r"_m\(([^)]+)\)", name)
    if m:
        out["method"] = m.group(1)
    m = re.search(r"_sz\(([^)]+)\)", name)
    if m:
        out["size"] = m.group(1)
    m = re.search(r"_a\(([^)]+)\)", name)
    if m:
        try:
            out["alpha_init"] = float(m.group(1))
        except ValueError:
            pass
    m = re.search(r"k\((\d+)\)", name)
    if m:
        out["k"] = int(m.group(1))
    return out


def merge_segments(segments: list[dict]) -> list[dict]:
    step_map = {}
    for seg in segments:
        for h in seg:
            step_map[h["step"]] = h
    return [step_map[s] for s in sorted(step_map)]


def choose_segment(segments: list[list[dict]]) -> list[dict]:
    """Pick the segment that reaches the farthest step, break ties by length."""
    return max(segments, key=lambda s: (s[-1]["step"], len(s)))


def align_series(base_steps: np.ndarray, steps: np.ndarray, values: np.ndarray) -> np.ndarray:
    value_map = {int(s): v for s, v in zip(steps, values)}
    return np.array([value_map.get(int(s), np.nan) for s in base_steps], dtype=float)


def last_finite(values):
    for v in reversed(values):
        if v is None:
            continue
        try:
            if np.isfinite(v):
                return float(v)
        except TypeError:
            continue
    return None


def last_non_null(hist_rows, key):
    for h in reversed(hist_rows):
        if key in h and h[key] is not None:
            return h[key]
    return None


# ----------------------------- set plots -----------------------------

def _best_alpha_from_1set() -> dict[tuple[str, str], float]:
    df = pd.read_csv(ONE_SET_BEST)
    best = {}
    for (method, size), g in df.groupby(["method", "size"]):
        g_non = g[g["alpha"] != 0.5]
        if not g_non.empty:
            idx = g_non["modes_mean"].idxmax()
            alpha = float(g.loc[idx, "alpha"])
        else:
            alpha = 0.5
        best[(method, size)] = alpha
    return best


def _best_alpha_from_2bit() -> dict[tuple[str, int], float]:
    df = pd.read_csv(TWO_BIT_BEST)
    best = {}
    for (method, k), g in df.groupby(["method", "k"]):
        g_non = g[g["alpha"] != 0.5]
        if not g_non.empty:
            idx = g_non["modes_mean"].idxmax()
            alpha = float(g.loc[idx, "alpha"])
        else:
            alpha = 0.5
        best[(method, int(k))] = alpha
    return best


def plot_set():
    records = json.loads((SET_JSON if SET_JSON.exists() else SET_JSON_FALLBACK).read_text())

    # group by (project, run_name)
    grouped = defaultdict(list)
    for rec in records:
        grouped[(rec.get("project"), rec.get("run_name"))].append(rec)

    runs = []
    for (project, run_name), recs in grouped.items():
        segments = []
        for rec in recs:
            hist = [h for h in rec.get("history", []) if isinstance(h.get("step"), (int, float))]
            if not hist:
                continue
            hist.sort(key=lambda x: x["step"])
            segments.append(hist)
        if not segments:
            continue

        if project == "Refactored-Alpha-GFN-Set-New-icml-fl0" and any(seg[0]["step"] > 0 for seg in segments):
            hist = merge_segments(segments)
        else:
            # For Rebuttal-Set-Temp-Old, prefer segments with reward_temp=1
            if project == "Rebuttal-Set-Temp-Old":
                filtered = []
                for seg in segments:
                    last = seg[-1]
                    if last.get("reward_temp") == 1:
                        filtered.append(seg)
                if filtered:
                    segments = filtered
            hist = choose_segment(segments)

        if not hist or hist[-1]["step"] < 9999:
            continue

        meta = parse_run_name(run_name or "")
        if "size" not in meta or "alpha_init" not in meta:
            continue

        last = hist[-1]
        if project == "Rebuttal-Set-Temp-Old":
            if last.get("reward_temp") != 1:
                continue
        record = {
            "size": meta["size"],
            "alpha_init": meta["alpha_init"],
            "method": meta.get("method"),
            "fl": last.get("fl"),
            "objective": last.get("objective") or meta.get("method"),
            "step": [h["step"] for h in hist],
            "project": project,
            "reward_temp": last.get("reward_temp"),
        }

        for metric in ["modes", "mean_top_1000_R", "spearman_corr_test", "loss", "forward_policy_entropy_eval"]:
            record[metric] = [h.get(metric) for h in hist]

        record["Objective"] = calculate_objective(record)
        runs.append(record)

    metrics_config = [
        {"metric": "modes", "title": r"$\text{Num Modes}$", "ylabel": "Number of Modes"},
        {"metric": "mean_top_1000_R", "title": r"$\text{Top-1000 Avg Reward}$", "ylabel": "Top-1000 Average Reward"},
        {"metric": "spearman_corr_test", "title": r"$\text{Spearman Corr}$", "ylabel": "Spearman Correlation"},
        {"metric": "loss", "title": r"$\text{Loss}$", "ylabel": "Loss"},
        {"metric": "forward_policy_entropy_eval", "title": r"$\text{Policy Entropy}$", "ylabel": "Policy Entropy"},
    ]

    objectives = ["DB", "FL-DB", "TB", "SubTB(λ)", "FL-SubTB(λ)"]
    best_map = _best_alpha_from_1set()

    def get_best_alpha(obj, size):
        method = {
            "DB": "db_gfn",
            "FL-DB": "fl_db_gfn",
            "TB": "tb_gfn",
            "SubTB(λ)": "subtb_gfn",
            "FL-SubTB(λ)": "fl_subtb_gfn",
        }[obj]
        return best_map[(method, size)]

    for sz in ["small", "medium", "large"]:
        for cfg in metrics_config:
            metric = cfg["metric"]
            ylabel = cfg["ylabel"]
            for obj in objectives:
                groups = defaultdict(list)
                best_alpha = get_best_alpha(obj, sz)

                # collect runs for this objective + size
                if obj == "FL-SubTB(λ)":
                    # Follow 1-set source mapping:
                    # - small/medium from Rebuttal-Set-FL with method fl_subtb_gfn
                    # - large from Refactored-Alpha-GFN-Set-New-icml with method subtb_gfn
                    if sz in ("small", "medium"):
                        cand = [
                            r
                            for r in runs
                            if r.get("project") == "Rebuttal-Set-FL"
                            and r.get("size") == sz
                            and r.get("method") == "fl_subtb_gfn"
                            and r.get("reward_temp") == 1
                        ]
                    else:
                        cand = [
                            r
                            for r in runs
                            if r.get("project") == "Refactored-Alpha-GFN-Set-New-icml"
                            and r.get("size") == sz
                            and r.get("method") == "subtb_gfn"
                            and r.get("reward_temp") == 1
                        ]
                else:
                    cand = [r for r in runs if r.get("Objective") == obj and r.get("size") == sz]
                if obj == "SubTB(λ)":
                    cand = [
                        r
                        for r in cand
                        if r.get("project") == "Refactored-Alpha-GFN-Set-New-icml-fl0"
                    ]
                if not cand:
                    continue

                # choose base steps from the longest run
                base = max(cand, key=lambda r: len(r["step"]))
                base_steps = np.array(base["step"], dtype=float)

                for r in cand:
                    steps = np.array(r["step"], dtype=float)
                    vals_raw = r.get(metric, [])
                    vals = np.array([v if v is not None else np.nan for v in vals_raw], dtype=float)
                    groups[r["alpha_init"]].append(align_series(base_steps, steps, vals))

                if 0.5 not in groups or best_alpha not in groups:
                    continue

                build_image(
                    base_steps,
                    groups,
                    ours_alpha=best_alpha,
                    title=obj,
                    ylabel=ylabel,
                    metric=metric,
                    x_range=[0, 10000],
                    ticks=[0, 2000, 4000, 6000, 8000],
                    tick_labels=["0", "2k", "4k", "6k", "8k"],
                    otick=9900,
                    otick_label="10k",
                    file_name=f"plot_set_{obj.replace('-','_').removesuffix('(λ)')}_alpha{best_alpha}_{sz}_{metric}",
                    save_dir=SET_SAVE,
                )


# ----------------------------- bit plots -----------------------------

def plot_bit():
    records = json.loads(BIT_JSON.read_text())

    grouped = defaultdict(list)
    for rec in records:
        grouped[(rec.get("project"), rec.get("run_name"))].append(rec)

    runs = []
    for (project, run_name), recs in grouped.items():
        segments = []
        for rec in recs:
            hist = [h for h in rec.get("history", []) if isinstance(h.get("step"), (int, float))]
            if not hist:
                continue
            hist.sort(key=lambda x: x["step"])
            segments.append(hist)
        if not segments:
            continue

        hist = choose_segment(segments)
        if not hist or hist[-1]["step"] < 49999:
            continue

        meta = parse_run_name(run_name or "")
        if "k" not in meta or "alpha_init" not in meta:
            continue

        last = hist[-1]
        use_exp_weight_decay = last_non_null(hist, "use_exp_weight_decay")
        if use_exp_weight_decay is not False:
            continue
        record = {
            "k": meta["k"],
            "alpha_init": meta["alpha_init"],
            "method": meta.get("method"),
            "fl": last.get("fl"),
            "objective": last.get("objective") or meta.get("method"),
            "step": [h["step"] for h in hist],
        }

        for metric in ["modes", "spearman_corr_test", "loss", "forward_policy_entropy_eval"]:
            record[metric] = [h.get(metric) for h in hist]

        record["Objective"] = calculate_objective(record)
        runs.append(record)

    metrics_config = [
        {"metric": "modes", "title": r"$\text{Num Modes}$", "ylabel": "Number of Modes"},
        {"metric": "spearman_corr_test", "title": r"$\text{Spearman Corr}$", "ylabel": "Spearman Correlation"},
        {"metric": "loss", "title": r"$\text{Loss}$", "ylabel": "Loss"},
        {"metric": "forward_policy_entropy_eval", "title": r"$\text{Policy Entropy}$", "ylabel": "Policy Entropy"},
    ]

    objective_order = ["DB", "SubTB(λ)", "TB", "FL-DB", "FL-SubTB(λ)", "FL-TB"]
    best_map = _best_alpha_from_2bit()

    def get_bit_best_alpha(obj, k):
        method_map = {
            "DB": "db",
            "SubTB(λ)": "subtb",
            "TB": "tb",
            "FL-DB": "fl-db",
            "FL-SubTB(λ)": "fl-subtb",
            "FL-TB": "fl-tb",
        }
        method = method_map.get(obj)
        if method is None:
            return None
        return best_map.get((method, k))

    objectives = [o for o in objective_order if o in {r.get("Objective") for r in runs}]
    k_values = sorted({r.get("k") for r in runs if r.get("k") is not None})

    for k in k_values:
        for cfg in metrics_config:
            metric = cfg["metric"]
            ylabel = cfg["ylabel"]
            for obj in objectives:
                groups = defaultdict(list)
                best_alpha = get_bit_best_alpha(obj, k)
                if best_alpha is None:
                    continue

                cand = [r for r in runs if r.get("Objective") == obj and r.get("k") == k]
                if not cand:
                    continue

                base = max(cand, key=lambda r: len(r["step"]))
                base_steps = np.array(base["step"], dtype=float)

                for r in cand:
                    steps = np.array(r["step"], dtype=float)
                    vals_raw = r.get(metric, [])
                    vals = np.array([v if v is not None else np.nan for v in vals_raw], dtype=float)
                    groups[r["alpha_init"]].append(align_series(base_steps, steps, vals))

                if 0.5 not in groups or best_alpha not in groups:
                    continue

                build_image(
                    base_steps,
                    groups,
                    ours_alpha=best_alpha,
                    title=obj,
                    ylabel=ylabel,
                    metric=metric,
                    file_name=f"plot_bit_{obj.replace('-','_').removesuffix('(λ)')}_alpha{best_alpha}_k{k}_{metric}",
                    save_dir=BIT_SAVE,
                )


# ----------------------------- mols plots -----------------------------


def _best_alpha_from_mols(metric: str = "num_modes_eval") -> dict[str, float]:
    records = json.loads(MOLS_JSON.read_text())
    scores = defaultdict(lambda: defaultdict(list))
    for rec in records.values():
        obj = calculate_objective(rec)
        if obj is None:
            continue
        alpha = rec.get("alpha_init")
        if alpha is None:
            continue
        values = rec.get(metric)
        if isinstance(values, list):
            val = last_finite(values)
        else:
            val = values
        if val is None:
            continue
        scores[obj][float(alpha)].append(float(val))

    best = {}
    for obj, amap in scores.items():
        non_base = {a: v for a, v in amap.items() if abs(a - 0.5) > 1e-8}
        cand = non_base or amap
        alpha = max(cand.items(), key=lambda kv: np.mean(kv[1]))[0]
        best[obj] = alpha
    return best


def plot_mols():
    records = json.loads(MOLS_JSON.read_text())
    runs = []
    for rec in records.values():
        steps = rec.get("step")
        if not steps or steps[-1] < 49999:
            continue
        record = {
            "alpha_init": rec.get("alpha_init"),
            "fl": rec.get("fl"),
            "objective": rec.get("objective"),
            "step": steps,
        }
        for metric in [
            "num_modes_eval",
            "top_100_avg_reward_eval",
            "top_100_avg_similarity_eval",
            "current_loss",
            "forward_policy_entropy_eval",
            "all_samples_avg_length_eval",
            "spearman_corr_test",
        ]:
            record[metric] = rec.get(metric)
        record["Objective"] = calculate_objective(record)
        runs.append(record)

    metrics_config = [
        {"metric": "num_modes_eval", "title": r"$\text{Num Modes}$", "ylabel": "Number of Modes"},
        {"metric": "top_100_avg_reward_eval", "title": r"$\text{Top-100 Avg Reward}$", "ylabel": "Top-100 Average Reward"},
        {
            "metric": "top_100_avg_similarity_eval",
            "title": r"$\text{Top-100 Avg Similarity}$",
            "ylabel": "Top-100 Average Similarity",
        },
        {"metric": "current_loss", "title": r"$\text{Loss}$", "ylabel": "Loss"},
        {"metric": "forward_policy_entropy_eval", "title": r"$\text{Policy Entropy}$", "ylabel": "Policy Entropy"},
        {"metric": "all_samples_avg_length_eval", "title": r"$\text{Avg Length}$", "ylabel": "Average Length"},
        {"metric": "spearman_corr_test", "title": r"$\text{Spearman Corr}$", "ylabel": "Spearman Correlation"},
    ]

    objectives = ["DB", "FL-DB", "SubTB(λ)", "FL-SubTB(λ)", "TB"]
    best_map = _best_alpha_from_mols(metric="num_modes_eval")
    best_map["TB"] = 0.6

    for cfg in metrics_config:
        metric = cfg["metric"]
        ylabel = cfg["ylabel"]
        for obj in objectives:
            groups = defaultdict(list)
            best_alpha = best_map.get(obj, 0.5)

            cand = [r for r in runs if r.get("Objective") == obj]
            if not cand:
                continue

            base = max(cand, key=lambda r: len(r["step"]))
            base_steps = np.array(base["step"], dtype=float)

            for r in cand:
                steps = np.array(r["step"], dtype=float)
                vals_raw = r.get(metric, [])
                vals = np.array([v if v is not None else np.nan for v in vals_raw], dtype=float)
                groups[r["alpha_init"]].append(align_series(base_steps, steps, vals))

            if 0.5 not in groups or best_alpha not in groups:
                continue

            build_image(
                base_steps,
                groups,
                ours_alpha=best_alpha,
                title=obj,
                ylabel=ylabel,
                metric=metric,
                x_range=[0, 50000],
                ticks=[0, 10000, 20000, 30000, 40000],
                tick_labels=["0", "10k", "20k", "30k", "40k"],
                otick=49900,
                otick_label="50k",
                file_name=f"plot_mols_{obj.replace('-','_').removesuffix('(λ)')}_alpha{best_alpha}_{metric}",
                save_dir=MOLS_SAVE,
            )


def main() -> None:
    have_any = False
    if SET_JSON.exists() or SET_JSON_FALLBACK.exists():
        have_any = True
        print("[INFO] Plotting set figures...")
        plot_set()
    else:
        print(f"[WARN] Missing {SET_JSON} and {SET_JSON_FALLBACK}; skip set plots")

    if BIT_JSON.exists():
        have_any = True
        print("[INFO] Plotting bit figures...")
        plot_bit()
    else:
        print(f"[WARN] Missing {BIT_JSON}; skip bit plots")

    if MOLS_JSON.exists():
        have_any = True
        print("[INFO] Plotting mols figures...")
        plot_mols()
    else:
        print(f"[WARN] Missing {MOLS_JSON}; skip mols plots")

    if not have_any:
        raise FileNotFoundError("No input JSON files found for plotting")
    print("[INFO] Done")


if __name__ == "__main__":
    main()
