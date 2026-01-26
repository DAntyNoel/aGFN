import json
import wandb

ENTITY = "1969773923-shanghai-jiao-tong-university"

PROJECTS = [
    "Rebuttal-Bit-FL",
    "Refactored-Alpha-GFN-Bitseq-icml2026",
    "Refactored-Alpha-GFN-Bitseq",
]

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

OUTPUT_JSON = "all_wandb_histories_merged.jsonl"


def summary_keys(histories, valid_keys):
    summarized = {}
    for his in histories:
        for key in valid_keys:
            if key not in summarized:
                summarized[key] = []
            summarized[key].append(his.get(key, "UNKNOWN"))
    for key in summarized:
        values = summarized[key]
        if values and all(v == values[0] for v in values[:-1]):
            summarized[key] = values[-1]
    return summarized


def fetch_history_with_retry(run, valid_keys, max_retries=5):
    for attempt in range(max_retries):
        history = run.history(pandas=False)
        if history and set(valid_keys).issubset(set(history[-1].keys())):
            return history
        print(
            f"⚠️ Run {run.name}: history incomplete, retry {attempt + 1}/{max_retries} ..."
        )
        return history
    raise RuntimeError(f"❌ Failed to fetch complete history for run {run.name}")


def main():
    api = wandb.Api(timeout=60)
    merged = []

    for project in PROJECTS:
        project_path = f"{ENTITY}/{project}"
        runs = api.runs(project_path)
        for run in runs:
            print(f"Processing project {project}: run {run.name}")
            try:
                history = fetch_history_with_retry(run, VALID_KEYS)
            except RuntimeError as e:
                print(str(e))
                continue
            summarized = summary_keys(history, VALID_KEYS)
            merged.append(
                {
                    "project": project,
                    "run_name": run.name,
                    "history": history,
                    "summary": summarized,
                }
            )

    with open(OUTPUT_JSON, "w") as f:
        json.dump(merged, f, indent=2)

    print(f"Saved merged histories to {OUTPUT_JSON}.")


if __name__ == "__main__":
    main()
