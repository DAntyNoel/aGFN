# PBP Alpha Diversity 0327

This folder mirrors the `1-set` workflow for a different W&B project:

- project: `1969773923-shanghai-jiao-tong-university/PBP-Alpha-Diversity-0327`
- download raw run summaries/configs
- parse metrics and map to target concepts:
  - `modes`
  - `spearman correlation`
  - `diversity`
  - `top rewards`

## Files

- `download_wandb.py`: download all runs and save CSV/JSON.
- `parse.py`: analyze metric keys and generate mapping + summary tables.
- `pbp_alpha_diversity_0327.csv`: raw downloaded table.
- `pbp_alpha_diversity_0327.json`: raw downloaded JSON records.
- `pbp_metric_inventory.csv`: all summary metric keys with availability stats.
- `pbp_metric_mapping.csv`: selected metric per target concept with candidate list.
- `pbp_target_candidates_overall.csv`: all related candidate metrics with overall `mean+-std`.
- `pbp_target_candidates_by_alpha.csv`: all related candidate metrics grouped by `alpha` with `mean+-std`.
- `pbp_target_selected_by_alpha.csv`: selected metric per target grouped by `alpha` with `mean+-std`.
- `pbp_target_candidates_plot.png`: 2x2 plot of all candidate metrics by target.
- `pbp_target_selected_plot.png`: 2x2 plot of selected metric per target.

## Run

```bash
cd /home/fhshao/aGFN/icml2026/pbp_alpha_diversity_0327
conda run -n agfn python download_wandb.py
conda run -n agfn python parse.py
```
