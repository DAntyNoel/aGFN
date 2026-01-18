import time
import wandb
import json

filters = {
    'step': 49999,
    'random_action_prob': 0.05,
    'use_exp_weight_decay': False
}

valid_keys = [
 'all_samples_avg_length_eval',
 'all_samples_avg_reward_eval',
 'alpha',
 'backward_policy_entropy_eval',
 'current_loss',
 'forward_policy_entropy_eval',
 'modes_avg_length_eval',
 'modes_avg_reward_eval',
 'modes_avg_similarity_eval',
 'num_modes_eval',
 'spearman_corr_test',
 'step',
 'top_1000_avg_reward_eval',
 'top_100_avg_reward_eval',
 'top_50_avg_reward_eval',
 'top_10_avg_reward_eval'
    # 'use_alpha_scheduler',
    # 'alpha_init',
    # 'forward_policy_entropy_test',
    # 'step',
    # 'loss',
    # 'mean_top_100_R',
    # 'mean_top_1000_R',
    # 'mean_R',
    # 'grad_clip_norm',
    # 'use_grad_clip',
    # 'modes',
    # 'method',
    # 'forward_policy_entropy_eval',
    # 'backward_policy_entropy_eval',
    # 'fl',
    # 'alpha',
    # 'spearman_corr_test'
]

sum_keys = [
 'alpha_init',
 'fl',
 'objective',
]

def summary_keys(historys, valid_keys, run_summary):
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
    # 添加 summary keys
    for key in sum_keys:
        if key in historys[-1]:
            summarized[key] = run_summary[key]
    return summarized


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
            yield history, run.name, run.summary

api = wandb.Api()
runs = api.runs("1969773923-shanghai-jiao-tong-university/Refactored-Alpha-GFN-Mols-New")
ttl = {}
for history, name, sum in set_filter_generator(runs, filters):
    print(f"Processing run: {name}")
    summarized = summary_keys(history, valid_keys, sum)
    ttl[name] = summarized
with open(f"mols_run_summary.json", "w") as f:
    json.dump(ttl, f, indent=2)
print(f"Collected summaries for {len(ttl)} runs.")