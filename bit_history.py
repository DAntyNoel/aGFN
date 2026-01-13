import time
import wandb
import json

filters = {
    'step': 49999,
    'grad_clip_norm': 20,
    'use_exp_weight_decay': False
}

valid_keys = [
 'alpha_init',
 'k',                             # k=4
 'M_size',                 # M_size (s,m,l)
 'objective',

 'step',
 'modes',                         # modes   
 'spearman_corr_test',              # spearman
 'loss',                           # loss
 'forward_policy_entropy_eval', # entropy

 'alpha',
#  'backward_policy_entropy_eval',
#  'backward_policy_entropy_test',
#  'forward_policy_entropy_test',
#  'mean_R_all_unique',
#  'mean_top_1000_R',
#  'mean_top_100_R',
#  'use_alpha_scheduler'
]

def summary_keys(historys, valid_keys):
    summarized = {}
    for his in historys:
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

def bit_history_with_retry(run, max_retries=5, delay=3):
    """带重试的 history 下载函数"""
    for attempt in range(max_retries):
        history = run.history(pandas=False)
        # 检查最后一个 step 是否包含全部 valid_keys
        if history and set(valid_keys).issubset(set(history[-1].keys())):
            return history
        print(f"⚠️ Run {run.name}: history incomplete, retry {attempt+1}/{max_retries} ...")
        time.sleep(delay)
    raise RuntimeError(f"❌ Failed to fetch complete history for run {run.name}")


def bit_filter_generator(runs, filters):
    for run in runs:
        match = True
        for key, value in filters.items():
            if key not in run.summary or run.summary[key] != value:
                match = False
                break
        if match:
            # history = bit_history_with_retry(run)
            history = run.history(pandas=False)
            yield history, run.name

api = wandb.Api(timeout=60)
runs = api.runs("1969773923-shanghai-jiao-tong-university/Refactored-Alpha-GFN-Bitseq")
ttl = {}
for history, name in bit_filter_generator(runs, filters):
    print(f"Processing run: {name}")
    summarized = summary_keys(history, valid_keys)
    ttl[name] = summarized
with open(f"bit_run_summary.json", "w") as f:
    json.dump(ttl, f, indent=2)
print(f"Collected summaries for {len(ttl)} runs.")