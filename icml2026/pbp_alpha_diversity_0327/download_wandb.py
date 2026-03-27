#!/usr/bin/env python3
"""Download runs from PBP-Alpha-Diversity-0327 and store raw run metadata.

Outputs:
- pbp_alpha_diversity_0327.csv
- pbp_alpha_diversity_0327.json
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import wandb

ENTITY = "1969773923-shanghai-jiao-tong-university"
PROJECT = "PBP-Alpha-Diversity-0327"

ROOT = Path(__file__).resolve().parent
OUT_CSV = ROOT / "pbp_alpha_diversity_0327.csv"
OUT_JSON = ROOT / "pbp_alpha_diversity_0327.json"


def clean_config(config: Dict[str, Any]) -> Dict[str, Any]:
    return {k: v for k, v in config.items() if not k.startswith("_")}


def main() -> None:
    api = wandb.Api()
    runs = api.runs(f"{ENTITY}/{PROJECT}")

    rows: List[Dict[str, Any]] = []
    records: List[Dict[str, Any]] = []

    for run in runs:
        summary = run.summary._json_dict
        config = clean_config(run.config)

        records.append(
            {
                "name": run.name,
                "id": run.id,
                "state": run.state,
                "created_at": str(run.created_at),
                "project": PROJECT,
                "entity": ENTITY,
                "config": config,
                "summary": summary,
            }
        )

        rows.append(
            {
                "name": run.name,
                "id": run.id,
                "state": run.state,
                "created_at": str(run.created_at),
                "project": PROJECT,
                "entity": ENTITY,
                "config": json.dumps(config, ensure_ascii=False),
                "summary": json.dumps(summary, ensure_ascii=False),
            }
        )

        print(f"downloaded: {run.name}")

    pd.DataFrame(rows).to_csv(OUT_CSV, index=False)
    with OUT_JSON.open("w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    print(f"saved csv: {OUT_CSV}")
    print(f"saved json: {OUT_JSON}")
    print(f"total runs: {len(rows)}")


if __name__ == "__main__":
    main()
