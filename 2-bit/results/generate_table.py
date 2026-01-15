import pandas as pd
import numpy as np

# 读取数据
df = pd.read_csv('data.csv')

# 按 k 和 method 找到 baseline (alpha=0.5) 和 ours (最高 modes)
data_dict = {}

for k in sorted(df['k'].unique()):
    for method in df['method'].unique():
        subset = df[(df['method'] == method) & (df['k'] == k)]
        if not subset.empty:
            # Baseline: alpha=0.5
            baseline_rows = subset[subset['alpha'] == 0.5]
            if not baseline_rows.empty:
                baseline_row = baseline_rows.iloc[0]
            else:
                baseline_row = None
            
            # Ours: 最高 modes_mean
            best_row = subset.loc[subset['alpha'] != 0.5]
            
            data_dict[(method, k)] = {
                'baseline': baseline_row,
                'ours': best_row
            }

# 生成新表格
print("\\begin{table}[htbp]")
print("\\caption{Results on Bit Sequence Generation. In terms of number of modes on average, $\\alpha$-GFN objectives outperform vanilla GFlowNet objectives across 87\\% task settings.}")
print("\\centering")
print("\\small")
print("\\setlength{\\tabcolsep}{6pt}")
print("\\renewcommand{\\arraystretch}{1.2}")
print("\\resizebox{0.99\\linewidth}{!}{%")
print("\\begin{tabular}{llcccccccccc}")
print("\\toprule")
print("\\multicolumn{1}{c}{} & \\multicolumn{1}{c}{} & \\multicolumn{2}{c}{\\textbf{DB}} & \\multicolumn{2}{c}{\\textbf{SubTB($\\lambda$)}} & \\multicolumn{2}{c}{\\textbf{TB}} & \\multicolumn{2}{c}{\\textbf{FL-DB}} & \\multicolumn{2}{c}{\\textbf{FL-SubTB}} \\\\")
print("\\cmidrule(lr){3-4}\\cmidrule(lr){5-6}\\cmidrule(lr){7-8}\\cmidrule(lr){9-10}\\cmidrule(lr){11-12}")
print("\\textbf{k} & \\textbf{Metric} & Baseline & Ours & Baseline & Ours & Baseline & Ours & Baseline & Ours & Baseline & Ours \\\\")
print("\\midrule")

k_values = sorted(df['k'].unique())
for idx, k in enumerate(k_values):
    # Modes 行
    print(f"\\multirow{{2}}{{*}}{{{k}}} &  Modes", end="")
    
    # DB
    if ('db', k) in data_dict:
        if data_dict[('db', k)]['baseline'] is not None:
            row = data_dict[('db', k)]['baseline']
            m = float(row['modes_mean'])
            s = float(row['modes_std'])
            print(f" & \\msl{{{m:.2f}}}{{{s:.2f}}}", end="")
        row = data_dict[('db', k)]['ours']
        m = float(row['modes_mean'])
        s = float(row['modes_std'])
        print(f" & \\msl{{\\textbf{{{m:.2f}}}}}{{{s:.2f}}}", end="")
    
    # SubTB
    if ('subtb', k) in data_dict:
        baseline_m = float(data_dict[('subtb', k)]['baseline']['modes_mean']) if data_dict[('subtb', k)]['baseline'] is not None else 0
        ours_m = float(data_dict[('subtb', k)]['ours']['modes_mean'])
        is_baseline_max = baseline_m >= ours_m
        
        if data_dict[('subtb', k)]['baseline'] is not None:
            row = data_dict[('subtb', k)]['baseline']
            m = float(row['modes_mean'])
            s = float(row['modes_std'])
            if is_baseline_max:
                print(f" & \\msl{{\\textbf{{{m:.2f}}}}}{{{s:.2f}}}", end="")
            else:
                print(f" & \\msl{{{m:.2f}}}{{{s:.2f}}}", end="")
        
        row = data_dict[('subtb', k)]['ours']
        m = float(row['modes_mean'])
        s = float(row['modes_std'])
        if not is_baseline_max:
            print(f" & \\msl{{\\textbf{{{m:.2f}}}}}{{{s:.2f}}}", end="")
        else:
            print(f" & \\msl{{{m:.2f}}}{{{s:.2f}}}", end="")
    
    # TB
    if ('tb', k) in data_dict:
        if data_dict[('tb', k)]['baseline'] is not None:
            row = data_dict[('tb', k)]['baseline']
            m = float(row['modes_mean'])
            s = float(row['modes_std'])
            print(f" & \\msl{{{m:.2f}}}{{{s:.2f}}}", end="")
        row = data_dict[('tb', k)]['ours']
        m = float(row['modes_mean'])
        s = float(row['modes_std'])
        print(f" & \\msl{{\\textbf{{{m:.2f}}}}}{{{s:.2f}}}", end="")
    
    # FL-DB
    if ('fl-db', k) in data_dict:
        if data_dict[('fl-db', k)]['baseline'] is not None:
            row = data_dict[('fl-db', k)]['baseline']
            m = float(row['modes_mean'])
            s = float(row['modes_std'])
            print(f" & \\msl{{{m:.2f}}}{{{s:.2f}}}", end="")
        row = data_dict[('fl-db', k)]['ours']
        m = float(row['modes_mean'])
        s = float(row['modes_std'])
        print(f" & \\msl{{\\textbf{{{m:.2f}}}}}{{{s:.2f}}}", end="")
    
    # FL-SubTB
    if ('fl-subtb', k) in data_dict:
        baseline_m = float(data_dict[('fl-subtb', k)]['baseline']['modes_mean']) if data_dict[('fl-subtb', k)]['baseline'] is not None else 0
        ours_m = float(data_dict[('fl-subtb', k)]['ours']['modes_mean'])
        
        if data_dict[('fl-subtb', k)]['baseline'] is not None:
            row = data_dict[('fl-subtb', k)]['baseline']
            m = float(row['modes_mean'])
            s = float(row['modes_std'])
            if baseline_m > ours_m:
                print(f" & \\msl{{\\textbf{{{m:.2f}}}}}{{{s:.2f}}}", end="")
            else:
                print(f" & \\msl{{{m:.2f}}}{{{s:.2f}}}", end="")
        
        row = data_dict[('fl-subtb', k)]['ours']
        m = float(row['modes_mean'])
        s = float(row['modes_std'])
        if not baseline_m > ours_m:
            print(f" & \\msl{{\\textbf{{{m:.2f}}}}}{{{s:.2f}}}", end="")
        else:
            print(f" & \\msl{{{m:.2f}}}{{{s:.2f}}}", end="")
    
    print(" \\\\")
    
    # Spearman 行
    print("&  Spearman", end="")
    
    # DB
    if ('db', k) in data_dict:
        if data_dict[('db', k)]['baseline'] is not None:
            row = data_dict[('db', k)]['baseline']
            m = float(row['spearman_corr_test_mean'])
            s = float(row['spearman_corr_test_std'])
            print(f" & \\msl{{{m:.2f}}}{{{s:.2f}}}", end="")
        row = data_dict[('db', k)]['ours']
        m = float(row['spearman_corr_test_mean'])
        s = float(row['spearman_corr_test_std'])
        print(f" & \\msl{{{m:.2f}}}{{{s:.2f}}}", end="")
    
    # SubTB
    if ('subtb', k) in data_dict:
        if data_dict[('subtb', k)]['baseline'] is not None:
            row = data_dict[('subtb', k)]['baseline']
            m = float(row['spearman_corr_test_mean'])
            s = float(row['spearman_corr_test_std'])
            print(f" & \\msl{{{m:.2f}}}{{{s:.2f}}}", end="")
        row = data_dict[('subtb', k)]['ours']
        m = float(row['spearman_corr_test_mean'])
        s = float(row['spearman_corr_test_std'])
        print(f" & \\msl{{{m:.2f}}}{{{s:.2f}}}", end="")
    
    # TB
    if ('tb', k) in data_dict:
        if data_dict[('tb', k)]['baseline'] is not None:
            row = data_dict[('tb', k)]['baseline']
            m = float(row['spearman_corr_test_mean'])
            s = float(row['spearman_corr_test_std'])
            print(f" & \\msl{{{m:.2f}}}{{{s:.2f}}}", end="")
        row = data_dict[('tb', k)]['ours']
        m = float(row['spearman_corr_test_mean'])
        s = float(row['spearman_corr_test_std'])
        print(f" & \\msl{{{m:.2f}}}{{{s:.2f}}}", end="")
    
    # FL-DB
    if ('fl-db', k) in data_dict:
        if data_dict[('fl-db', k)]['baseline'] is not None:
            row = data_dict[('fl-db', k)]['baseline']
            m = float(row['spearman_corr_test_mean'])
            s = float(row['spearman_corr_test_std'])
            print(f" & \\msl{{{m:.2f}}}{{{s:.2f}}}", end="")
        row = data_dict[('fl-db', k)]['ours']
        m = float(row['spearman_corr_test_mean'])
        s = float(row['spearman_corr_test_std'])
        print(f" & \\msl{{{m:.2f}}}{{{s:.2f}}}", end="")
    
    # FL-SubTB
    if ('fl-subtb', k) in data_dict:
        if data_dict[('fl-subtb', k)]['baseline'] is not None:
            row = data_dict[('fl-subtb', k)]['baseline']
            m = float(row['spearman_corr_test_mean'])
            s = float(row['spearman_corr_test_std'])
            print(f" & \\msl{{{m:.2f}}}{{{s:.2f}}}", end="")
        row = data_dict[('fl-subtb', k)]['ours']
        m = float(row['spearman_corr_test_mean'])
        s = float(row['spearman_corr_test_std'])
        print(f" & \\msl{{{m:.2f}}}{{{s:.2f}}}", end="")
    
    if idx < len(k_values) - 1:
        print(" \\\\")
        print("\\midrule")
    else:
        print(" \\\\")

print("\\bottomrule")
print("\\end{tabular}")
print("}% end resizebox")
print("\\label{tab:bit-exp-results}")
print("% \\vspace{-1.5\\baselineskip}")
print("\\end{table}")
