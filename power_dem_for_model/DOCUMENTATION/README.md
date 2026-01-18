# Power Demand Forecasting Pipeline

## 📋 Project Description

**Power Demand Forecasting** is a time-series analysis and forecasting system for electricity consumption data from the **LD2011_2014 dataset** (UK household meters). The pipeline uses **SARIMAX** and **Prophet** models across three forecasting strategies (ALL_METER, CATEGORY, INDIVIDUAL) to predict electricity demand for various horizons (30-180 days).

### Key Features:
- 🔄 Complete ETL pipeline (extract, transform, load)
- 📊 Multi-strategy forecasting (aggregate, seasonal, individual)
- 🤖 Dual algorithm support (SARIMAX + Prophet)
- 📈 Automatic hyperparameter tuning
- 📉 Comprehensive evaluation metrics (sMAPE, R²)
- 💾 Model serialization and Excel export
- 📝 Jupyter notebook orchestration

### Dataset:
- **Source:** LD2011_2014.txt (semicolon-delimited CSV)
- **Period:** January 2011 - December 2014 (4 years)
- **Frequency:** Hourly readings
- **Meters:** ~370 household electricity meters

---

## 🚀 Quick Start (5 minutes)

### Step 1: Clone from Git
```powershell
# Navigate to your workspace directory
cd C:\Users\[YourUsername]\OneDrive\Documents

# Clone the repository
git clone https://github.com/your-repo/power_demand_forecasting.git
cd power_demand_forecasting

# OR manually download and extract the folder
```

### Step 2: Set Up Environment
See **⚙️ Environment Setup** section below to create environment and install dependencies from `requirements.txt`.

Once activated, continue:

### Step 3: Run the Pipeline
```powershell
# Start Jupyter notebook
jupyter notebook notebook/main_notebook.ipynb

# Execute all cells (Kernel > Run All)
```

### Step 4: View Results
```powershell
# Open results in Excel
start results/model_results/evaluation_metrics_testing.xlsx
```

---

## 📂 Project Folder Structure

```
power_dem_for_model/
│
├── 📚 DOCUMENTATION (Start here!)
│   ├── README.md                      ← You are here
│   ├── QUICK_REFERENCE.md             ← Quick-start guide
│   ├── CODE_DOCUMENTATION.md          ← Implementation details
│
├── 🔧 CONFIGURATION
│   ├── config/
│   │   └── config.py                  ← All paths defined here
│   └── requirements.txt                ← Python dependencies
│
├── 💻 SOURCE CODE (Well-Documented)
│   └── src/
│       ├── data_cleaning.py           ← Load & clean raw data
│       ├── feature_selection.py       ← Engineer features
│       ├── model_build.py             ← Train models
│       ├── eval_matrix.py             ← Export results
│       └── pred_visualization.py      ← Visualize forecasts
│
├── 📓 NOTEBOOKS
│   └── notebook/
│       ├── main_notebook.ipynb        ← Main execution pipeline

├── 📁 DATA DIRECTORIES
│   └── data/
│       ├── raw_data/
│       │   └── LD2011_2014.txt       ← Raw electricity meter data
│       └── processed_data/            ← Output from data_cleaning
│           ├── df_hourly.csv
│           ├── df_daily.csv
│           ├── meter_summary_df.csv
│           └── problematic_df.csv
│
└── 📈 RESULTS DIRECTORIES
    └── results/
        ├── Models/                    ← Trained models (.joblib)
        ├── model_results/             ← Evaluation metrics (Excel)
        │   └── evaluation_metrics_testing.xlsx
        └── pred/                      ← Forecasts & visualizations
            └── future_prediction_*.xlsx
```

---

## ⚙️ Environment Setup (Step-by-Step)

### Using requirements.txt (Recommended)

This is the easiest and most reliable way to set up your environment. The `requirements.txt` file contains all necessary Python packages.

**Option 1: Using Conda (Best for Windows)**

```powershell
# Step 1: Create conda environment
conda create -n powerenv python=3.9 -y

# Step 2: Activate environment
conda activate powerenv

# Step 3: Install all packages from requirements.txt
pip install -r requirements.txt

# Step 4: Verify installation
python -c "from prophet import Prophet; import pandas; print('✅ All packages installed!')"
```

**Option 2: Using Virtual Environment (venv)**

```powershell
# Step 1: Navigate to project folder
cd C:\Users\...\power_dem_for_model

# Step 2: Create virtual environment
python -m venv .venv

# Step 3: Activate virtual environment
.\.venv\Scripts\Activate.ps1

# If activation fails, run:
# Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

# Step 4: Upgrade pip
python -m pip install --upgrade pip

# Step 5: Install all packages from requirements.txt
pip install -r requirements.txt

# Step 6: Verify installation
python -c "from prophet import Prophet; import pandas; print('✅ All packages installed!')"
```

**Deactivate Environment:**
```powershell
deactivate
```

---

## 🔄 Cloning from Git (Step-by-Step)

### Prerequisites:
- Git installed on your machine
- GitHub account (if private repository)

### Steps:

```powershell
# Step 1: Open PowerShell and navigate to workspace
cd C:\Users\[YourUsername]\OneDrive\Documents

# Step 2: Clone repository
git clone https://github.com/your-org/power-demand-forecasting.git

# Step 3: Navigate into project
cd power-demand-forecasting

# Step 4: Verify folder structure
dir
# Expected: config/, data/, notebook/, src/, results/, requirements.txt, README.md
```

### After Cloning - Set Up Environment:

See **⚙️ Environment Setup** section below.

---

## 📖 Getting Started After Setup

Once your environment is set up (see **⚙️ Environment Setup** above), follow these steps:

### 1. Activate Environment
```powershell
conda activate powerenv
```

### 2. Open Jupyter Notebook
```powershell
# Start Jupyter in the notebook folder
jupyter notebook notebook/main_notebook.ipynb
```

### 3. Run Pipeline
```
In Jupyter:
1. Kernel > Run All (or execute cell by cell)
2. Monitor progress in output cells
3. Results save to results/ folder automatically
```

### 4. View Results
```powershell
# Excel results
start results/model_results/evaluation_metrics_testing.xlsx

# Forecast file
start results/pred/future_prediction_35d.xlsx

# Check saved models
dir results/Models/
```

---

## ✅ Verification Checklist

After setup, verify everything works:

```powershell
# ✅ Check Python version
python --version
# Expected: Python 3.9.x

# ✅ Check environment is active
conda info --envs
# Expected: powerenv marked with *

# ✅ Check imports
python -c "import pandas, numpy, prophet, statsmodels, sklearn; print('✅ All OK')"

# ✅ Check data file exists
Test-Path data/raw_data/LD2011_2014.txt
# Expected: True

# ✅ Check Jupyter runs
jupyter --version
# Expected: Version number displayed

# ✅ Check config paths
python -c "from config.config import RAW_FILE, RESULTS_DIR; print(f'Raw: {RAW_FILE}'); print(f'Results: {RESULTS_DIR}')"
# Expected: Paths printed successfully
```

---

## 🐛 Troubleshooting

### Issue: Prophet Installation Fails
```powershell
# Solution 1: Use conda-forge
conda install -c conda-forge prophet -y

# Solution 2: If conda fails, try conda-forge with Python 3.10
conda create -n powerenv python=3.10 -y
conda activate powerenv
conda install -c conda-forge prophet -y
```

### Issue: Jupyter Not Found
```powershell
# Solution: Install jupyter
conda install -c conda-forge jupyter -y

# Or with pip
pip install jupyter
```

### Issue: Data File Not Found
```powershell
# Check existence
Test-Path data/raw_data/LD2011_2014.txt

# If missing, ensure it's downloaded:
# 1. Download from source
# 2. Place in data/raw_data/ folder
# 3. Verify filename: LD2011_2014.txt (case-sensitive on some systems)
```

### Issue: ModuleNotFoundError
```powershell
# Solution: Reinstall all packages
pip install --upgrade --force-reinstall -r requirements.txt

# Or with conda
conda install --force-reinstall -c conda-forge -r requirements.txt -y
```

### Issue: PowerShell Execution Policy
```powershell
# If .venv\Scripts\Activate.ps1 fails:
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

# Then try activation again:
.\.venv\Scripts\Activate.ps1
```

---

## 📚 Documentation Files

| File | Purpose | Read Time |
|------|---------|-----------|
| **QUICK_REFERENCE.md** | Quick-start & common tasks | 5-10 min |
| **API_REFERENCE.md** | Function documentation | 10-15 min |
| **CODE_DOCUMENTATION.md** | Implementation details | 15-20 min |
| **DOCUMENTATION_INDEX.md** | Navigation guide | 5 min |

**Start with:** QUICK_REFERENCE.md

---

## 🎯 Common Commands

```powershell
# Activate environment
conda activate powerenv

# Deactivate environment
conda deactivate

# View conda environments
conda info --envs

# Update all packages
conda update --all -y

# Run Jupyter notebook
jupyter notebook notebook/main_notebook.ipynb

# Check Python packages
pip list

# Install specific version
pip install pandas==1.5.3
```


## 📝 Notes

- **Python Version:** 3.9+ recommended
- **OS:** Developed on Windows (PowerShell)
- **Database:** Uses CSV files (no database needed)
- **GPU:** Not required (CPU is sufficient)
- **RAM:** 4GB+ recommended
- **Storage:** 500MB+ for data and results

---

## 📞 Support

- Check **QUICK_REFERENCE.md** for quick answers
- Review **API_REFERENCE.md** for function details
- See **CODE_DOCUMENTATION.md** for implementation
- Visit **DOCUMENTATION_INDEX.md** for navigation

---

## 🎉 You're Ready!

Your environment is set up and ready to use. Start with:

```powershell
conda activate powerenv
jupyter notebook notebook/main_notebook.ipynb
```

Happy forecasting! 📊

---

**Last Updated:** January 18, 2026  
**Version:** 1.0  
**Status:** ✅ Production Ready
