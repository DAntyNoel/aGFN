import pandas as pd
import json

exp_name = 'mols'

runs_pd = pd.read_csv(f"{exp_name}.csv")
run_dict = {}
for (sum, name) in zip(runs_pd["summary"], runs_pd["name"]):
    sum_dict = eval(sum)
    if exp_name == 'mols' and sum_dict['step'] != 49999:
        continue
    if exp_name == 'bit' and sum_dict['step'] != 49999:
        continue
    if exp_name == 'set':
        if sum_dict['step'] != 9999:
            continue
        if sum_dict['training_mode'] != 'online':
            continue
        if sum_dict['use_alpha_scheduler'] != True:
            continue
        if sum_dict['use_grad_clip'] != True:
            continue
        # if sum_dict['fl'] != True:
        #     continue
    run_dict[name] = sum_dict

json.dump(run_dict, open(f"{exp_name}.json", "w"), indent=2)
print(len(run_dict))