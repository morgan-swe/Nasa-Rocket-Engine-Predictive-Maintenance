## Predictive Maintenance for Rotating Industrial Equipment

This project builds a machine learning pipeline to predict the Remaining Useful Life (RUL) of turbofan engines using multivariate sensor time-series data (NASA C-MAPSS FD001 dataset) — the same class of problem energy infrastructure companies face when monitoring compressors, pumps, and turbines along a pipeline network.

**The problem:** Unplanned failures of rotating equipment are expensive and, in energy infrastructure, can pose safety and environmental risk. Rather than relying on fixed maintenance schedules or reacting after a failure, this project explores how continuous sensor telemetry (temperature, pressure, vibration-adjacent measurements) can be used to forecast how many operating cycles remain before a unit degrades to failure — enabling maintenance to be scheduled proactively, only when it's actually needed.

### Results

Three models were trained on 80 engines and evaluated on 20 held-out engines the models had never seen. Error is measured in operating cycles, against an average engine lifetime of roughly 206 cycles.

| Model | MAE (cycles) | RMSE (cycles) | R² |
|---|---|---|---|
| Linear Regression (baseline) | 14.17 | 17.65 | 0.821 |
| Decision Tree | 11.63 | 17.11 | 0.832 |
| **Random Forest** | **10.38** | **15.20** | **0.867** |

The Random Forest predicts remaining life to within about 10 cycles on average — roughly 5% of a typical engine's lifespan, and a 27% reduction in error over the linear baseline.

### What was built

**Exploratory analysis**
- Profiled 100 engine units and 21 onboard sensors across 20,631 cycle records
- Sensor triage by variance: identified flat/uninformative channels to exclude, and used a correlation matrix to find redundant sensors measuring the same underlying physical process
- Characterized the engine lifetime distribution (right-skewed, most failures between 160–240 cycles)

**Feature engineering**
- Constructed the RUL target from cycle-to-failure data
- Generated 63 time-series features per row — rolling means, rolling standard deviations, and cycle-over-cycle deltas for all 21 sensors:
  - **Rolling means** separate long-term degradation trends from short-term sensor noise
  - **Rolling standard deviations** capture growing instability, which often precedes a shift in the average value — the classic early-warning signature on rotating equipment
  - **Delta features** capture the rate of change in a sensor, which can signal failure before absolute values look abnormal

**Modeling**
- Split by engine unit using `GroupShuffleSplit` so no engine appears in both train and test — a random row-level split would leak cycles from the same engine across the boundary and inflate accuracy
- Capped RUL at 125 cycles, since sensor data carries little degradation signal early in an engine's life
- Trained and compared linear regression, decision tree, and random forest regressors
- Evaluated with MAE, RMSE, and R², plus prediction-vs-actual scatter and residual plots for each model
- Serialized trained models to `models/` with joblib

### Project structure

```
data/raw/         Original NASA C-MAPSS FD001 text files
data/interim/     Parsed and validated CSVs
data/processed/   Feature-engineered dataset (91 columns)
notebooks/        Analysis, versioned stage by stage
models/           Serialized trained models (.pkl)
reports/figures/  Saved plots
src/              Data loading and config
```

Notebooks:
- `01_data_overview.ipynb` — dataset profiling, sensor variance, correlation analysis
- `05_feature_engineering.ipynb` — RUL target, rolling and trend features, train/test split
- `06_model_training.ipynb` — model training, evaluation, and comparison

### Tech stack

Python, pandas, NumPy, scikit-learn, matplotlib, seaborn, Jupyter

### Status and next steps

The analysis pipeline currently lives in the notebooks; `src/load_data.py` handles raw data loading and validation. Planned next:
- Refactor the notebook logic into the `src/` modules (`features.py`, `preprocess.py`, `train.py`, `evaluate.py`) so the pipeline runs end to end from the command line
- Feature importance analysis to identify which sensors drive the predictions
- Hyperparameter tuning and cross-validation across engine groups
- Streamlit dashboard for loading a saved model and viewing per-engine RUL predictions
