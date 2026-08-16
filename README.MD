## Predictive Maintenance for Rotating Industrial Equipment

This project builds a machine learning pipeline to predict the Remaining Useful Life (RUL) of turbofan engines using multivariate sensor time-series data (NASA C-MAPSS FD001 dataset) — the same class of problem energy infrastructure companies face when monitoring compressors, pumps, and turbines along a pipeline network.

The problem: Unplanned failures of rotating equipment are expensive and, in energy infrastructure, can pose safety and environmental risk. Rather than relying on fixed maintenance schedules or reacting after a failure, this project explores how continuous sensor telemetry (temperature, pressure, vibration-adjacent measurements) can be used to forecast how many operating cycles remain before a unit degrades to failure — enabling maintenance to be scheduled proactively, only when it's actually needed.

What was built:
- Exploratory data analysis across 100+ engine units and 21 onboard sensors to identify which signals carry meaningful degradation information versus which are flat/uninformative (a real-world sensor triage problem)
- Engineered a full time-series feature set from raw sensor streams per unit, including:
  - Rolling means and rolling standard deviations to separate long-term degradation trends from short-term noise/instability
  - Cycle-over-cycle trend (delta) features to detect the rate of change in a sensor, which often signals failure earlier than the raw value drifting
- Constructed the RUL target variable from cycle-to-failure data, the label the eventual regression/ML model is trained to predict
- Structured as a reproducible pipeline (src/) separating data loading, feature engineering, preprocessing, and modeling stages, with exploratory work versioned notebook-by-notebook

Tech stack: Python, pandas, NumPy, matplotlib, scikit-learn (upcoming modeling stage)
