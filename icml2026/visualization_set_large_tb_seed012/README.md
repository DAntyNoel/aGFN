# Set Large TB Visualization (sampling_seed = 0/1/2)

This folder follows the filtering logic of `1-set/parse.py` and focuses on:
- `size == large`
- `method == tb_gfn`
- `sampling_seed in {0,1,2}`

## Files

- `build_set_large_tb_seed012.py`: extract rows and generate summary/plot.
- `set_large_tb_seed012_raw.csv`: matched raw runs.
- `set_large_tb_seed012_summary.csv`: grouped mean/std by `alpha_init`.
- `set_large_tb_seed012_plot.png`: 2x2 metric plot (generated if `matplotlib` exists).

## Run

```bash
cd /home/fhshao/aGFN/icml2026/visualization_set_large_tb_seed012
python build_set_large_tb_seed012.py
```
