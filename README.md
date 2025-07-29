
# EnergyAI‑TimeSeriesLab

**End‑to‑end deep‑learning pipeline for smart‑building power data**  
Data cleaning ▶ feature engineering ▶ LSTM/CNN‑LSTM/Transformer ▶ anomaly detection ▶ transfer learning.

---

## 🔥 Key Data‑Science Highlights


| Skill | How it’s demonstrated here |
|-------|---------------------------|
| **Time‑series feature engineering** | Cyclical encodings (`sin/cos`), holiday/weekend flags, Isolation‑Forest outlier capping, hour‑level resampling. |
| **Model development at scale** | LSTM, CNN‑LSTM, and Transformer architectures **Bayesian‑tuned** with Keras Tuner. |
| **Rigorous evaluation** | RMSE, MAE, R², **CV‑RMSE**, multi‑horizon error curves, hour‑of‑day error maps. |
| **Transfer learning** | 72 % RMSE drop by fine‑tuning the champion model on a low‑data floor. |
| **Anomaly analytics** | Mean + 3 σ threshold, 51 anomalies characterised by hour & weekday distributions. |
| **Domain‑specific KPIs** | EUI (205.4 kWh m⁻² yr⁻¹), load factor, peak/base loads, daily archetype profiles. |
| **Reproducibility** | Global `np`/`tf` seeds, Conda `environment.yml`, exact hyper‑params logged, notebook + HTML render. |

---

## 📂 Repository structure
```text
EnergyAI-TimeSeriesLab/
├── EnergyAI-TimeSeriesLab.ipynb   # main notebook (fully explained, runnable)
├── EnergyAI-TimeSeriesLab.html    # static render for quick browsing
├── data/                          # (add Floor1‑7 CSVs here, .gitignored for size)
├── environment.yml                # conda spec for Apple‑Silicon + TensorFlow‑Metal
└── README.md
```

---

## 🚀 Quick start
```bash
# 1️⃣  Clone + set up env
git clone https://github.com/mnikoopayan/EnergyAI-TimeSeriesLab.git
cd EnergyAI-TimeSeriesLab
conda env create -f environment.yml
conda activate tf_m1      # same env used in the study

# 2️⃣  Drop the seven Floor*.csv files into ./data
#     (Dataset originally part of the CU‑BEMS public release.)

# 3️⃣  Launch the lab
jupyter lab EnergyAI-TimeSeriesLab.ipynb
```

All code blocks are **cell‑by‑cell runnable**; each section is self‑contained so you can jump straight to modelling or anomaly analytics.

---

## 📊 Key results

### One‑step‑ahead forecasting (test set, Floor 6)
| Model | RMSE (kWh) | MAE (kWh) | R² | CV‑RMSE (%) | Train time |
|-------|-----------:|----------:|---:|------------:|-----------:|
| **Tuned LSTM** | **3.62** | 2.17 | 0.986 | 16.5 | 203 s |
| CNN‑LSTM | 3.71 | 2.18 | 0.985 | 16.9 | 314 s |
| Transformer | 4.29 | 2.42 | 0.981 | 19.6 | 422 s |

### Multi‑horizon champion (tuned LSTM)
| Horizon | RMSE | MAE | R² | NMAE % |
|---------|-----:|----:|---:|-------:|
| 1 h | 4.22 | 2.59 | 0.981 | 11.9 |
| 3 h | 4.89 | 2.92 | 0.975 | 13.4 |
| 6 h | 4.92 | 3.02 | 0.974 | 13.8 |
| 12 h | 6.71 | 3.56 | 0.952 | 16.4 |
| 24 h | 8.13 | 4.00 | 0.929 | 18.4 |

### Transfer learning (Floor 4, low‑data July 2018)
| Strategy | Test RMSE |
|----------|----------:|
| Train from scratch | 20.78 kWh |
| **Fine‑tuned champion** | **5.76 kWh** |

> **Δ = ‑72 %** error — showcases how pretrained sequence encoders cut data requirements.

---

## 🧐 Project story
1. **Data wrangling** – 689 128 rows × 30 sensor channels (per floor) cleaned; power → energy via hourly means.  
2. **Feature factory** – time & cyclical features, holiday flags, Isolation‑Forest capping at 99.5 % quantile.  
3. **Model zoo** – three deep‑learning families, **Bayesian‑optimised** then fully trained with early‑stopping.  
4. **Champion selection** – LSTM wins on RMSE + compute cost; extended to multi‑output forecaster.  
5. **Explainability hooks** – predictions‑vs‑actuals plots, error heat‑maps, anomaly visualisations.  
6. **Cross‑floor transfer** – demonstrate model reuse when Phase‑II deployment data are scarce.  
7. **Building KPIs** – EUI, load factor, weekday/weekend archetypes for facility ops teams.

---

## 📌 Requirements
* Python ≥ 3.10  
* TensorFlow 2.16 + Metal for Apple Silicon (`environment.yml`)  
* keras‑tuner, scikit‑learn, seaborn, pandas, matplotlib  

---

## ▶️ Next milestones
* Integrate weather covariates (dry‑bulb, dew‑point) via API pull.  
* Serve forecasts through FastAPI + Streamlit dashboard.  
* Extend transfer‑learning demo to unseen buildings (domain adaptation).

---

## ✨ Author
**Mohammad Saleh Nikoopayan Tak** – PhD candidate @ NJIT | Urban Systems Tech | Data‑Science for the Built Environment  
[LinkedIn](https://www.linkedin.com/in/nikoopayan) • [GitHub](https://github.com/mnikoopayan)

*Open to collaborations and conversations on smart‑building data science!*
