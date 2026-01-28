import json
from pathlib import Path
import wandb
from tqdm import tqdm

ENTITY = "1969773923-shanghai-jiao-tong-university"
PROJECT = "Refactored-Alpha-GFN-Bitseq"
OUTPUT = Path("6-appd/refactored_alpha_gfn_bitseq_histories.jsonl")

VALID_KEYS = [
    "alpha_init",
    "k",
    "M_size",
    "objective",
    "step",
    "modes",
    "spearman_corr_test",
    "loss",
    "forward_policy_entropy_eval",
    "alpha",
]


def summary_keys(histories, valid_keys):
    summarized = {}
    for his in histories:
        for key in valid_keys:
            summarized.setdefault(key, []).append(his.get(key, "UNKNOWN"))
    for key in list(summarized):
        values = summarized[key]
        if values and all(v == values[0] for v in values[:-1]):
            summarized[key] = values[-1]
    return summarized


def fetch_history_with_retry(run, valid_keys, max_retries=5):
    for attempt in range(max_retries):
        history = run.history(pandas=False)
        if history and set(valid_keys).issubset(set(history[-1].keys())):
            return history
        print(f"⚠️ Run {run.name}: history incomplete, retry {attempt + 1}/{max_retries} ...")
    return history


def load_seen(output_path: Path) -> set[str]:
    seen = set()
    if not output_path.exists():
        return seen
    with output_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                break
            name = obj.get("run_name")
            if name:
                seen.add(name)
    return seen


def main():
    api = wandb.Api(timeout=60)
    project_path = f"{ENTITY}/{PROJECT}"
    runs = list(api.runs(project_path))
    total = len(runs)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    seen = load_seen(OUTPUT)
    if seen:
        print(f"Resuming: {len(seen)} runs already in {OUTPUT}")
    print(f"Total runs: {total}, pending: {max(total - len(seen), 0)}")

    with OUTPUT.open("a") as f:
        written = 0
        skipped = 0
        with tqdm(runs, total=total, unit="run", desc="Downloading") as pbar:
            for idx, run in enumerate(pbar, 1):
                if run.name in seen:
                    skipped += 1
                    if skipped % 25 == 0:
                        pbar.set_postfix_str(f"skip {run.name}")
                    continue
                pbar.set_postfix_str(f"dl {run.name}")
                try:
                    history = fetch_history_with_retry(run, VALID_KEYS)
                except Exception as exc:
                    tqdm.write(f"⚠️ Failed run {run.name}: {exc}")
                    continue
                summarized = summary_keys(history, VALID_KEYS)
                obj = {
                    "project": PROJECT,
                    "run_name": run.name,
                    "history": history,
                    "summary": summarized,
                }
                f.write(json.dumps(obj))
                f.write("\n")
                written += 1

    print(f"Saved to {OUTPUT} (new runs: {written})")


if __name__ == "__main__":
    main()
