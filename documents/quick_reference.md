# Quick Reference Guide - Power Demand Forecasting

## 🚀 Quick Start

### Step 1: Run the Complete Pipeline
```python
# Open: power_dem_for_model/notebook/main_notebook.ipynb
# Execute all cells from top to bottom
```

### Step 2: Adjust Forecast Horizons (Optional)
```python
# In main_notebook.ipynb, cell 3:
HORIZONS = [30] ## Keep only single value for which span you want prediction
```

### Step 3: View Results
- **Models:** `results/Models/*.joblib` (trained models)
- **Metrics:** `results/model_results/evaluation_metrics_testing.xlsx`
- **Forecasts:** `results/pred/future_prediction_*.xlsx`

---

## 📊 Module Functions

### data_cleaning.py
```python
from src.data_cleaning import run_data_cleaning

df_hourly, df_daily, problematic_df, meter_summary = run_data_cleaning(
    raw_file=RAW_FILE,
    proc_folder=PROC_FOLDER
)
```
**Returns:** Cleaned hourly/daily data + meter statistics

---

### feature_selection.py
```python
from src.feature_selection import build_time_features

df_features, season_cols, month_avg = build_time_features(
    df_daily=df_daily,
    save=False  # Set True to persist features
)
```
**Returns:** Enhanced dataframe with temporal features + season columns

---

### model_build.py
```python
from src.model_build import run_strategies

results = run_strategies(
    df_daily=df_features,
    meter_summary_df=meter_summary,
    season_cols=season_cols,
    horizons=[30, 60, 90]
)
```
**Returns:** DataFrame with model performance metrics

---

### eval_matrix.py
```python
from src.eval_matrix import save_results

output_file = save_results(results_df)
```
**Returns:** Path to Excel file with results (4 sheets)

---

### pred_visualization.py
```python
from src.pred_visualization import run_prediction_visualization

run_prediction_visualization(
    df_daily=df_features,
    season_cols=season_cols,
    requested_horizon=35
)
```
**Outputs:** 3 plots + Excel predictions file

---

## 🔧 Configuration

All paths defined in `config/config.py`:
```python
from config.config import (
    RAW_FILE,        # LD2011_2014.txt
    PROC_FOLDER,     # data/processed_data/
    RESULTS_DIR,     # results/
    MODELS_DIR,      # results/Models/
    MODEL_RESULTS_DIR,  # results/model_results/
    PRED_DIR         # results/pred/
)
```

---

## 📈 Model Strategies

### 1. ALL_METER
- Single model on total demand
- **Algorithms:** SARIMAX (9 variants) + Prophet (3 configs)

### 2. CATEGORY
- Separate models for normal vs outlier meters
- Threshold: 99th percentile of daily average load
- **Algorithm:** Prophet (3 configurations)

### 3. INDIVIDUAL
- Normal meters aggregated + individual models for outliers
- **Algorithm:** Prophet (1 configuration)

---

## 📊 Performance Metrics

| Metric | Description | Good Range |
|--------|-------------|-----------|
| **sMAPE (%)** | Symmetric Mean Absolute % Error | 0-20% ✅ |
| **R²** | Coefficient of Determination | 0.8-1.0 ✅ |

Lower sMAPE = Better Forecasts
Higher R² = Better Model Fit

---

## 📁 Output Files

### Data Processing Outputs
```
data/processed_data/
├── df_hourly.csv           # Hourly data (2,191,680 rows)
├── df_daily.csv            # Daily data with features (1,459 rows)
├── meter_summary_df.csv     # Per-meter statistics
└── problematic_df.csv       # Meters with issues
```

### Model Outputs
```
results/
├── Models/
│   ├── final_all_meter_sarimax_35d.joblib
│   ├── final_all_meter_prophet_35d.joblib
│   ├── final_category_prophet_35d.joblib
│   └── final_individual_prophet_35d.joblib
├── model_results/
│   └── evaluation_metrics_testing.xlsx (4 sheets)
└── pred/
    └── future_prediction_35d.xlsx
```

---

## 🎯 Common Tasks

### Task: Change Forecast Horizon
```python
# In main_notebook.ipynb, cell 3:
HORIZONS = [45]  # Change from your period
```

### Task: Save Feature-Enhanced Data
```python
# In main_notebook.ipynb, cell 2:
df_daily_feat, season_cols, month_avg = build_time_features(
    df_daily,
    save=True,  # Enable saving
    proc_folder=PROC_FOLDER
)
```

### Task: Load a Trained Model
```python
import joblib
from pathlib import Path

model_path = Path("results/Models/final_all_meter_prophet_35d.joblib")
model = joblib.load(model_path)
# Use model.forecast() or model.predict() as needed
```

### Task: Analyze Results
```python
import pandas as pd

results = pd.read_excel(
    "results/model_results/evaluation_metrics_testing.xlsx",
    sheet_name="BEST_BY_HORIZON"
)
print(results)
```

---

## ⚠️ Common Issues & Solutions

### Issue: Excel File Not Found
```
Error: FileNotFoundError: [Errno 2] No such file or directory: 
'results/model_results/evaluation_metrics_testing.xlsx'
```
**Solution:** Run `save_results()` first to generate the file

---

### Issue: Prophet Installation Failed
```
Error: ModuleNotFoundError: No module named 'prophet'
```
**Solution:** Install using conda-forge
```bash
conda install -c conda-forge prophet
```

---

### Issue: Insufficient Data
```
Error: ValueError: ufunc 'isfinite' not supported for the input types
```
**Solution:** Ensure forecast horizon ≤ available data days (usually ~1,459)

---

### Issue: Missing Season Columns
```
Error: KeyError: 'season_category_normal'
```
**Solution:** Run `build_time_features()` before model building

---

## 📞 Support Information

**Data Source:** LD2011_2014 (UK electricity meters)  
**Time Period:** 2011-2014 (4 years)  
**Data Frequency:** Hourly  
**Number of Meters:** 370  
**Time Zone:** Greenwich Mean Time (GMT)

---

## 🔍 Key Metrics Explained

### Symmetric Mean Absolute Percentage Error (sMAPE)
```
sMAPE = (200/n) × Σ|actual - predicted| / (|actual| + |predicted|)
Scale: 0% (perfect) to 100% (very poor)
```

### R² Score (Coefficient of Determination)
```
R² = 1 - (SS_res / SS_tot)
Range: 0.0 (poor) to 1.0 (perfect)
Interpretation: % variance explained by model
```

---

**Last Updated:** January 18, 2026  
**Version:** 1.0
