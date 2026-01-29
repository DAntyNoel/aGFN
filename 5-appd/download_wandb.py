import json
import time
import wandb

ENTITY = "1969773923-shanghai-jiao-tong-university"

PROJECTS = [
    "Rebuttal-Set-Temp-Old",
    "Refactored-Alpha-GFN-Set-New-icml",
    "Refactored-Alpha-GFN-Set-New-icml-fl0",
    "Rebuttal-Set-FL",
]

FILTERS_BY_PROJECT = {
    "Rebuttal-Set-Temp-Old": {
        "step": 9999,
        "training_mode": "online",
        "use_alpha_scheduler": True,
        "use_grad_clip": False,
        "reward_temp": 1,
    },
    "Refactored-Alpha-GFN-Set-New-icml": {
        "step": 9999,
        "training_mode": "online",
        "use_alpha_scheduler": True,
        "use_grad_clip": False,
        "reward_temp": 1,
        "fl": True,
    },
    "Refactored-Alpha-GFN-Set-New-icml-fl0": {},
    "Rebuttal-Set-FL": {
        "step": 9999,
        "training_mode": "online",
        "use_alpha_scheduler": True,
        "use_grad_clip": False,
        "reward_temp": 1,
    },
}

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

OUTPUT_JSON = "wandb_histories_merged.json"


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


def fetch_history_with_retry(run, valid_keys, max_retries=5, delay=3):
    for attempt in range(max_retries):
        history = run.history(pandas=False)
        if history and set(valid_keys).issubset(set(history[-1].keys())):
            return history
        print(
            f"⚠️ Run {run.name}: history incomplete, retry {attempt + 1}/{max_retries} ..."
        )
        time.sleep(delay)
    raise RuntimeError(f"❌ Failed to fetch complete history for run {run.name}")


def set_filter_generator(runs, filters):
    for run in runs:
        match = True
        for key, value in filters.items():
            if key not in run.summary or run.summary[key] != value:
                match = False
                break
        if match:
            yield run


def main():
    api = wandb.Api()
    merged = {}
    cnt = 0

    with open(OUTPUT_JSON, "w") as f:
        for project in PROJECTS:
            project_path = f"{ENTITY}/{project}"
            runs = api.runs(project_path)
            filters = FILTERS_BY_PROJECT.get(project, {})
            merged[project] = {}
            for run in set_filter_generator(runs, filters):
                # print(f"Processing project {project}: run {run.name}")
                history = fetch_history_with_retry(run, VALID_KEYS, 2, 1)
                summarized = summary_keys(history, VALID_KEYS)
                merged[project][str(cnt) + '-' + run.name] = summarized
                cnt += 1
                print(f"✅ Processed run {run.name}")
            print(f"Finished processing project {project}.")
        json.dump(merged, f, indent=2)
    print(f"Saved merged histories to {OUTPUT_JSON}.")


if __name__ == "__main__":
    main()