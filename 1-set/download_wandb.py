import pandas as pd
import wandb
api = wandb.Api()

def d1():
    # Project is specified by <entity/project-name>
    runs = api.runs("1969773923-shanghai-jiao-tong-university/Rebuttal-Set-Temp-Old")

    summary_list, config_list, name_list = [], [], []
    for run in runs:
        # .summary contains the output keys/values for metrics like accuracy.
        #  We call ._json_dict to omit large files
        summary_list.append(run.summary._json_dict)

        # .config contains the hyperparameters.
        #  We remove special values that start with _.
        config_list.append(
            {k: v for k,v in run.config.items()
            if not k.startswith('_')})

        # .name is the human-readable name of the run.
        name_list.append(run.name)
        print(run.name)

    runs_df = pd.DataFrame({
        "summary": summary_list,
        "config": config_list,
        "name": name_list
        })
        
    runs_df.to_csv("rebuttal_set_temp_old.csv")

def d2():
    # Project is specified by <entity/project-name>
    runs = api.runs("1969773923-shanghai-jiao-tong-university/Refactored-Alpha-GFN-Set-New-icml")

    summary_list, config_list, name_list = [], [], []
    for run in runs:
        # .summary contains the output keys/values for metrics like accuracy.
        #  We call ._json_dict to omit large files
        summary_list.append(run.summary._json_dict)

        # .config contains the hyperparameters.
        #  We remove special values that start with _.
        config_list.append(
            {k: v for k,v in run.config.items()
            if not k.startswith('_')})

        # .name is the human-readable name of the run.
        name_list.append(run.name)
        print(run.name)

    runs_df = pd.DataFrame({
        "summary": summary_list,
        "config": config_list,
        "name": name_list
        })
        
    runs_df.to_csv("refactored_alpha_gfn_set_new_icml.csv")

def d3():
    # Project is specified by <entity/project-name>
    runs = api.runs("1969773923-shanghai-jiao-tong-university/Refactored-Alpha-GFN-Set-New-icml-fl0")

    summary_list, config_list, name_list = [], [], []
    for run in runs:
        # .summary contains the output keys/values for metrics like accuracy.
        #  We call ._json_dict to omit large files
        summary_list.append(run.summary._json_dict)

        # .config contains the hyperparameters.
        #  We remove special values that start with _.
        config_list.append(
            {k: v for k,v in run.config.items()
            if not k.startswith('_')})

        # .name is the human-readable name of the run.
        name_list.append(run.name)
        print(run.name)

    runs_df = pd.DataFrame({
        "summary": summary_list,
        "config": config_list,
        "name": name_list
        })
        
    runs_df.to_csv("refactored_alpha_gfn_set_new_icml_fl0.csv")

def d4():
    # Project is specified by <entity/project-name>
    runs = api.runs("1969773923-shanghai-jiao-tong-university/Rebuttal-Set-FL")

    summary_list, config_list, name_list = [], [], []
    for run in runs:
        # .summary contains the output keys/values for metrics like accuracy.
        #  We call ._json_dict to omit large files
        summary_list.append(run.summary._json_dict)

        # .config contains the hyperparameters.
        #  We remove special values that start with _.
        config_list.append(
            {k: v for k,v in run.config.items()
            if not k.startswith('_')})

        # .name is the human-readable name of the run.
        name_list.append(run.name)
        print(run.name)

    runs_df = pd.DataFrame({
        "summary": summary_list,
        "config": config_list,
        "name": name_list
        })
        
    runs_df.to_csv("rebuttal_set_fl.csv")

if __name__ == '__main__':
    # d1()
    # d2()
    d3()
    # d4()