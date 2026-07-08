# CU-BEMS Probabilistic Building Energy Forecasting

Reproducibility repository for the manuscript:

**An Explainable Probabilistic Deep Learning Framework for Building Energy
Forecasting: Multi Horizon Prediction, Anomaly Detection, and Few Shot Transfer
Learning**

This repository contains the analysis notebook, reusable helper code, saved
metrics, and figure artifacts used for the CU-BEMS building energy forecasting
study. The workflow benchmarks probabilistic LSTM, CNN-LSTM, and Transformer
models; selects a CNN-LSTM champion; extends it to multi-horizon quantile
forecasting; screens candidate operational anomalies with prediction intervals;
uses SHAP for local and global explanation; and evaluates strict few-shot
cross-floor transfer from Floor 6 to Floor 4.

## Repository Contents

```text
CU-BEMS-Probabilistic-Building-Energy-Forecasting/
├── data/
│   ├── README.md
│   └── raw/                         # CU-BEMS Floor1-Floor7 CSVs via Git LFS
├── notebooks/
│   └── cu_bems_probabilistic_building_energy_forecasting.ipynb
├── outputs/
│   ├── figures/                     # saved analysis figures
│   ├── figures/manuscript/          # final manuscript and composite figures
│   ├── metrics/                     # saved CSV metrics used in tables
│   └── README.md
├── scripts/
│   └── run_few_shot_protocol_search.py
├── src/
│   └── utils.py
├── requirements.txt
└── README.md
```

## Data

The study uses the CU-BEMS dataset collected from Chamchuri 5 at Chulalongkorn
University. The raw floor-level CSV files are tracked in `data/raw/` with Git
LFS. If the CSV files do not download automatically after cloning, run:

```bash
git lfs install
git lfs pull
```

Dataset citation:

Pipattanasomporn, M. et al. CU-BEMS, smart building electricity consumption and
indoor environmental sensor datasets. *Scientific Data* 7, 241 (2020).
https://doi.org/10.1038/s41597-020-00582-3

## Environment

The original profiling was performed on an Apple M1 system with 8 GB unified
memory using TensorFlow-Metal and Metal Performance Shaders acceleration. The
code also runs on CPU or standard TensorFlow GPU environments, although timing
results will be hardware dependent.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Reproducing the Analysis

Run the main notebook from the repository root:

```bash
jupyter lab notebooks/cu_bems_probabilistic_building_energy_forecasting.ipynb
```

The strict few-shot transfer search is intentionally separated into a script
because it is computationally expensive:

```bash
python scripts/run_few_shot_protocol_search.py
```

Saved outputs are already provided for reproducibility checks:

- `outputs/metrics/probabilistic_one_step_five_seed_summary.csv`
- `outputs/metrics/probabilistic_one_step_five_seed_per_seed.csv`
- `outputs/metrics/probabilistic_multi_horizon_metrics.csv`
- `outputs/metrics/probabilistic_anomalies_summary.csv`
- `outputs/metrics/strict_best_recipe_test_summary.csv`
- `outputs/metrics/strict_best_recipe_per_window_test.csv`
- `outputs/metrics/compute_profiling.csv`

## Manuscript-Aligned Results

### One-step benchmark on Floor 6

The final manuscript reports the neural models as mean (SD) over five fixed
architecture random-seed runs. Zero-training baselines are reported separately
in `outputs/metrics/zero_training_baselines.csv`.

| Model | Mean pinball loss (kWh) | RMSE (kWh) | MAE (kWh) | R-squared | CV(RMSE) (%) | PICP (%) | MPIW (kWh) |
|---|---:|---:|---:|---:|---:|---:|---:|
| LSTM | 0.613 (0.026) | 3.982 (0.148) | 2.293 (0.108) | 0.9830 (0.0013) | 17.91 (0.67) | 86.66 (1.00) | 8.42 (0.52) |
| CNN-LSTM | 0.561 (0.021) | 3.753 (0.198) | 2.062 (0.079) | 0.9849 (0.0016) | 16.88 (0.89) | 90.09 (0.81) | 9.36 (0.47) |
| Transformer | 0.813 (0.145) | 4.814 (0.883) | 2.945 (0.557) | 0.9746 (0.0094) | 21.66 (3.97) | 95.74 (2.30) | 17.33 (4.19) |

### Multi-horizon CNN-LSTM forecasting

| Horizon | RMSE (kWh) | MAE (kWh) | R-squared | PICP (%) | MPIW (kWh) |
|---:|---:|---:|---:|---:|---:|
| 1 h | 4.372 | 2.552 | 0.9794 | 95.29 | 15.67 |
| 3 h | 4.535 | 2.752 | 0.9779 | 95.53 | 16.34 |
| 6 h | 4.581 | 2.661 | 0.9774 | 95.47 | 17.14 |
| 12 h | 5.910 | 2.961 | 0.9624 | 96.48 | 21.17 |
| 24 h | 6.367 | 3.040 | 0.9561 | 96.42 | 22.25 |

### Candidate anomaly screening

The q05 to q95 interval rule identified 132 candidate operational anomalies in
the Floor 6 test period. These are review candidates, not confirmed faults.
The supporting summary is in
`outputs/metrics/probabilistic_anomalies_summary.csv`.

### Strict few-shot transfer from Floor 6 to Floor 4

The strict transfer experiment evaluates ten non-overlapping Floor 4 windows
for each labeled-data budget and compares scratch training with transfer
initialization from the Floor 6 CNN-LSTM champion.

| Training budget | Windows | Scratch RMSE mean (kWh) | Transfer RMSE mean (kWh) | Paired improvement mean (kWh) |
|---|---:|---:|---:|---:|
| 1 day | 10 | 35.12 | 24.45 | 10.67 |
| 3 days | 10 | 41.12 | 21.38 | 19.74 |
| 7 days | 10 | 36.17 | 18.17 | 18.01 |
| 14 days | 10 | 26.71 | 17.17 | 9.54 |
| All windows | 40 | 34.78 | 20.29 | 14.49 |

## Notes on Scope

The repository is intended to support manuscript reproducibility. It does not
claim that interval exceedances are confirmed equipment faults, and the SHAP
analysis explains the total-load model rather than separate end-use channels.
Raw CU-BEMS AC, lighting, and plug-load channels are available in the floor CSV
files for post-hoc operational review and future channel-level modeling.

## License and Attribution

The CU-BEMS dataset is distributed by its original authors under the terms
stated with the dataset publication. Please cite the CU-BEMS Scientific Data
paper when using the data. Code and derived analysis artifacts in this
repository are provided for research reproducibility.
