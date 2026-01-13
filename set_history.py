import time
import wandb
import json

filters = {
    'step': 9999,
    'training_mode': 'online',
    'use_alpha_scheduler': True,
    'use_grad_clip': False,
    'use_exp_weight_decay': 1,
    'alpha_warm_frac': 0.9,
    'exp_weight': 0.05
}

valid_keys = [
 'alpha',
 'alpha_init',
 'method',
 'fl',
 'size',               # size (s,m,l)
 'step',

 'modes',                           # modes
 'mean_top_1000_R',                # mean_top_1000_R
 'spearman_corr_test',          # spearman
 'loss',                           # loss
 'forward_policy_entropy_eval',  # entropy
#  'alpha_train_first_steps',
#  'backward_avg_policy_entropy_test',
#  'backward_policy_entropy_eval',
#  'forward_policy_entropy_test',
#  'mean_R',
#  'mean_top_100_R',
#  'n_train_steps',
]

def summary_keys(historys, valid_keys):
    summarized = {}
    for his in historys:
        present_keys = set(his.keys())
        # assert set(valid_keys).issubset(present_keys), f"Some valid keys are missing: {set(valid_keys) - present_keys}"
        for key in valid_keys:
            if key not in summarized:
                summarized[key] = []
            summarized[key].append(his.get(key, 'UNKNOWN'))
    # 检查是否有固定值
    for key in summarized:
        values = summarized[key]
        if all(v == values[0] for v in values[:-1]):
            summarized[key] = values[-1]
    return summarized

def fetch_history_with_retry(run, max_retries=5, delay=3):
    """带重试的 history 下载函数"""
    for attempt in range(max_retries):
        history = run.history(pandas=False)
        # 检查最后一个 step 是否包含全部 valid_keys
        if history and set(valid_keys).issubset(set(history[-1].keys())):
            return history
        print(f"⚠️ Run {run.name}: history incomplete, retry {attempt+1}/{max_retries} ...")
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
            # history = fetch_history_with_retry(run)
            history = run.history(pandas=False)
            yield history, run.name

api = wandb.Api()
runs = api.runs("1969773923-shanghai-jiao-tong-university/Refactored-Alpha-GFN-Set-New")
ttl = {}
for history, name in set_filter_generator(runs, filters):
    print(f"Processing run: {name}")
    summarized = summary_keys(history, valid_keys)
    ttl[name] = summarized
with open(f"set_run_summary.json", "w") as f:
    json.dump(ttl, f, indent=2)
print(f"Collected summaries for {len(ttl)} runs.")