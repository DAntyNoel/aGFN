#!/usr/bin/env python3
"""Merge resumed runs (same run_name) by step into a new dataset.

This expects the input file to be a JSON list produced by download_all.py,
even if the filename ends with .jsonl.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def _merge_histories(segments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    step_map: dict[int, dict[str, Any]] = {}
    for seg in segments:
        for h in seg["history"]:
            step = h.get("step")
            if isinstance(step, (int, float)):
                step_map[int(step)] = h
    return [step_map[s] for s in sorted(step_map)]


def merge_project_records(records: list[dict[str, Any]], project: str) -> list[dict[str, Any]]:
    by_run: dict[str, list[tuple[int, dict[str, Any]]]] = defaultdict(list)
    for idx, rec in enumerate(records):
        if rec.get("project") == project:
            run_name = rec.get("run_name") or ""
            by_run[run_name].append((idx, rec))

    merged_by_run: dict[str, dict[str, Any]] = {}
    for run_name, items in by_run.items():
        segments: list[dict[str, Any]] = []
        for pos, rec in items:
            hist = [h for h in rec.get("history", []) if isinstance(h.get("step"), (int, float))]
            if not hist:
                continue
            hist.sort(key=lambda x: x["step"])
            segments.append(
                {
                    "pos": pos,
                    "min_step": hist[0]["step"],
                    "max_step": hist[-1]["step"],
                    "history": hist,
                    "rec": rec,
                }
            )

        if not segments:
            continue

        # Later segments override earlier ones. Order by min_step, then by file position.
        segments.sort(key=lambda s: (s["min_step"], s["pos"]))
        merged_hist = _merge_histories(segments)

        # Keep metadata from the segment that reached the farthest step.
        best = max(segments, key=lambda s: s["max_step"])
        base = best["rec"]
        merged_rec = {k: v for k, v in base.items() if k != "history"}
        merged_rec["history"] = merged_hist
        merged_by_run[run_name] = merged_rec

    # Rebuild list in original order, replacing duplicates with the merged record.
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for rec in records:
        if rec.get("project") != project:
            out.append(rec)
            continue
        run_name = rec.get("run_name") or ""
        if run_name in seen:
            continue
        merged = merged_by_run.get(run_name)
        if merged is not None:
            out.append(merged)
        seen.add(run_name)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge resumed runs by step.")
    parser.add_argument(
        "--input",
        default="5-appd/all_wandb_histories_merged.jsonl",
        help="Input JSON list from download_all.py",
    )
    parser.add_argument(
        "--output",
        default="5-appd/all_wandb_histories_merged_resumed.json",
        help="Output JSON list with merged runs",
    )
    parser.add_argument(
        "--project",
        default="Refactored-Alpha-GFN-Set-New-icml-fl0",
        help="Project name to merge",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    records = json.loads(input_path.read_text())

    merged = merge_project_records(records, args.project)
    output_path.write_text(json.dumps(merged, indent=2))

    print(f"[INFO] Input records: {len(records)}")
    print(f"[INFO] Output records: {len(merged)}")
    print(f"[INFO] Wrote: {output_path}")


if __name__ == "__main__":
    main()
