import pandas as pd
import wandb
import json
import os
from pathlib import Path

api = wandb.Api()

def download_with_history(project_name, output_prefix):
    """
    下载 wandb project 的数据并提取 history（时间序列数据）
    根据 parse.py 中的筛选条件进行过滤
    返回格式：{run_name: {config信息 + history数据}}
    """
    print(f"\n=== 下载 {project_name} ===")
    runs = api.runs(f"1969773923-shanghai-jiao-tong-university/{project_name}")
    
    run_dict = {}
    skipped = 0
    for idx, run in enumerate(runs):
        try:
            run_name = run.name
            
            # 获取配置
            config = {k: v for k, v in run.config.items() if not k.startswith('_')}
            
            # 获取最终 summary
            summary = run.summary._json_dict
            
            # === 应用 parse.py 中的筛选条件 ===
            should_skip = False
            
            # 所有项目都需要的筛选
            if summary.get('step') != 9999:
                should_skip = True
            if summary.get('training_mode') != 'online':
                should_skip = True
            if summary.get('use_alpha_scheduler') != True:
                should_skip = True
            if summary.get('use_grad_clip') != False:
                should_skip = True
            if summary.get('reward_temp') != 1:
                should_skip = True
            
            # 项目特定的筛选
            if project_name == 'Refactored-Alpha-GFN-Set-New-icml':
                if summary.get('fl') != False:
                    should_skip = True
            elif project_name == 'Refactored-Alpha-GFN-Set-New-icml-fl0':
                if summary.get('fl') != False:
                    should_skip = True
            elif project_name == 'Rebuttal-Set-FL':
                # 不筛选 fl 字段，包含所有
                pass
            
            if should_skip:
                skipped += 1
                continue
            
            print(f"{idx+1}. {run_name}", end=" ... ")
            
            # 获取 history（时间序列数据）
            history = run.history()  # 返回 DataFrame
            
            # 整合数据
            record = {
                **config,
                **summary,
                'step': history['step'].tolist() if 'step' in history.columns else [],
                'loss': history['loss'].tolist() if 'loss' in history.columns else [],
                'modes': history['modes'].tolist() if 'modes' in history.columns else [],
                'mean_top_1000_R': history.get('mean_top_1000_R', history.get('reward', pd.Series([]))).tolist() if any(col in history.columns for col in ['mean_top_1000_R', 'reward']) else [],
                'spearman_corr_test': history.get('spearman_corr_test', pd.Series([])).tolist() if 'spearman_corr_test' in history.columns else [],
                'forward_policy_entropy_eval': history.get('forward_policy_entropy_eval', pd.Series([])).tolist() if 'forward_policy_entropy_eval' in history.columns else [],
            }
            
            run_dict[run_name] = record
            print("✓")
            
        except Exception as e:
            print(f"✗ {e}")
            continue
    
    print(f"✓ 成功下载 {len(run_dict)} 个 runs（跳过 {skipped} 个不符合条件的 runs）")
    return run_dict

def download_history_and_merge():
    """
    下载三个实验的 history 数据并合并成一个 JSON
    """
    print("\n" + "="*80)
    print("下载并合并所有实验的 History 数据")
    print("="*80)
    
    # 下载三个项目的数据
    data1 = download_with_history(
        "Refactored-Alpha-GFN-Set-New-icml",
        "refactored_alpha_gfn_set_new_icml"
    )
    
    data2 = download_with_history(
        "Refactored-Alpha-GFN-Set-New-icml-fl0",
        "refactored_alpha_gfn_set_new_icml_fl0"
    )
    
    data3 = download_with_history(
        "Rebuttal-Set-FL",
        "rebuttal_set_fl"
    )
    
    # 合并三个数据源
    merged_data = {}
    merged_data.update(data1)
    merged_data.update(data2)
    merged_data.update(data3)
    
    # 保存为 JSON（用于 test4.ipynb）
    output_file = "set_run_summary_with_history.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(merged_data, f, indent=2)
    
    print(f"\n✓ 已保存合并后的数据到：{output_file}")
    print(f"✓ 共合并 {len(merged_data)} 个 runs")
    
    # 打印数据格式示例
    if merged_data:
        first_key = list(merged_data.keys())[0]
        first_record = merged_data[first_key]
        print(f"\n数据格式示例（{first_key}）：")
        print(f"  keys: {list(first_record.keys())}")
        if 'step' in first_record and first_record['step']:
            print(f"  step 数据点数: {len(first_record['step'])}")
        if 'loss' in first_record and first_record['loss']:
            print(f"  loss 数据点数: {len(first_record['loss'])}")

if __name__ == '__main__':
    # 下载 history 数据并合并成 JSON（用于画图）
    download_history_and_merge()