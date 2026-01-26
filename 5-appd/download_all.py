import json
import wandb

ENTITY = "1969773923-shanghai-jiao-tong-university"

PROJECTS = [
    "Rebuttal-Set-Temp-Old",
    "Refactored-Alpha-GFN-Set-New-icml",
    "Refactored-Alpha-GFN-Set-New-icml-fl0",
    "Rebuttal-Set-FL",
]

VALID_KEYS = [
    "alpha",
    "alpha_init",
    "method",
    "fl",
    "size",
    "step",
    "modes",
    "mean_top_1000_R",
    "spearman_corr_test",
    "loss",
    "forward_policy_entropy_eval",
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


def fetch_history_with_retry(run, valid_keys, max_retries=1):
    for attempt in range(max_retries):
        history = run.history(pandas=False)
        if history and set(valid_keys).issubset(set(history[-1].keys())):
            return history
        print(
            f"⚠️ Run {run.name}: history incomplete, retry {attempt + 1}/{max_retries} ..."
        )
        return history


def main():
    api = wandb.Api()
    merged = []

    for project in PROJECTS:
        project_path = f"{ENTITY}/{project}"
        runs = api.runs(project_path)
        for run in runs:
            print(f"Processing project {project}: run {run.name}")
            history = fetch_history_with_retry(run, VALID_KEYS)
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