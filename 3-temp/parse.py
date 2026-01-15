import pandas as pd
import json


def d1():
    exp_name = 'rebuttal_set_temp_fixed'

    runs_pd = pd.read_csv(f"{exp_name}.csv")
    run_dict = {}
    failed = 0
    success = 0
    cnt = 0
    for (sum, name) in zip(runs_pd["summary"], runs_pd["name"]):
        sum_dict = eval(sum)
        # print(sum_dict.keys())
        cnt += 1
        # exit()
        try:
            if exp_name == 'rebuttal_set_temp_fixed':
                if sum_dict['step'] != 9999:
                    continue
                if sum_dict['training_mode'] != 'online':
                    continue
                if sum_dict['use_alpha_scheduler'] != True:
                    continue
                if sum_dict['use_grad_clip'] != False:
                    continue
                # if sum_dict['reward_temp'] != 1:
                #     continue
                # if sum_dict['fl'] != True:
                #     continue
            run_dict[name] = sum_dict
            success += 1
        except KeyError as e:
            failed += 1
            print(f"KeyError for run {name}: {e}")
            print(f"Summary keys: {sum_dict.keys()}")

    json.dump(run_dict, open(f"{exp_name}.json", "w"), indent=2)
    print(f"Total {cnt}. Successful parses: {success}, Failed parses: {failed}, length of run_dict: {len(run_dict)}")


if __name__ == '__main__':
    d1()