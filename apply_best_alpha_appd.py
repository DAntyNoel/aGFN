#!/usr/bin/env python3
"""Apply best-alpha selections from 1-set / 2-bit to 5-appd / 6-appd.

Outputs:
- 5-appd/results/data_best_alpha_from_1set.csv
- 6-appd/results/data_best_alpha_from_2bit.csv
"""
from __future__ import annotations

from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent

ONE_SET = ROOT / "1-set" / "results" / "data.csv"
TWO_BIT = ROOT / "2-bit" / "results" / "data.csv"
APPD_SET = ROOT / "5-appd" / "results" / "data.csv"
APPD_BIT = ROOT / "6-appd" / "results" / "data.csv"

OUT_SET = ROOT / "5-appd" / "results" / "data_best_alpha_from_1set.csv"
OUT_BIT = ROOT / "6-appd" / "results" / "data_best_alpha_from_2bit.csv"

EPS = 1e-9


def pick_best_alpha_1set(df: pd.DataFrame) -> dict[tuple[str, str], float]:
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


def pick_best_alpha_2bit(df: pd.DataFrame) -> dict[tuple[str, int], float]:
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


def filter_appd(df: pd.DataFrame, best_map: dict, key_cols: list[str], include_baseline: bool = True) -> pd.DataFrame:
    keep_rows = []
    for _, row in df.iterrows():
        key = tuple(row[c] for c in key_cols)
        best = best_map.get(key)
        if best is None:
            continue
        alpha = float(row["alpha"])
        if include_baseline and abs(alpha - 0.5) < EPS:
            keep_rows.append(row)
            continue
        if abs(alpha - best) < EPS:
            keep_rows.append(row)
    if not keep_rows:
        return pd.DataFrame(columns=df.columns)
    return pd.DataFrame(keep_rows).sort_values(key_cols + ["alpha"]).reset_index(drop=True)


def main() -> None:
    df1 = pd.read_csv(ONE_SET)
    df2 = pd.read_csv(TWO_BIT)
    df5 = pd.read_csv(APPD_SET)
    df6 = pd.read_csv(APPD_BIT)

    best_1set = pick_best_alpha_1set(df1)
    best_2bit = pick_best_alpha_2bit(df2)

    filtered_5 = filter_appd(df5, best_1set, ["method", "size"], include_baseline=True)
    filtered_6 = filter_appd(df6, best_2bit, ["method", "k"], include_baseline=True)

    OUT_SET.write_text(filtered_5.to_csv(index=False))
    OUT_BIT.write_text(filtered_6.to_csv(index=False))

    print(f"Wrote {OUT_SET} ({len(filtered_5)} rows)")
    print(f"Wrote {OUT_BIT} ({len(filtered_6)} rows)")


if __name__ == "__main__":
    main()
