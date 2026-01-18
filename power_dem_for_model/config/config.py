"""
================================================================================
PROJECT CONFIGURATION MODULE
================================================================================

Purpose:
    Centralized configuration for all file paths and directories used across
    the power demand forecasting pipeline.

Key Features:
    - Dynamic path resolution (portable across machines)
    - Automatic directory creation
    - Single source of truth for project structure

Author: Power Demand Forecasting Team
Version: 1.0
================================================================================
"""

from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# BASE DIRECTORY RESOLUTION
# ─────────────────────────────────────────────────────────────────────────────
# This file location: power_dem_for_model/config/config.py
# So parents[1] = power_dem_for_model (parent of config/)
BASE_DIR = Path(__file__).resolve().parents[1]

# ─────────────────────────────────────────────────────────────────────────────
# DATA DIRECTORIES
# ─────────────────────────────────────────────────────────────────────────────
# Raw data folder: power_dem_for_model/data/raw_data/
RAW_FOLDER = BASE_DIR / "data" / "raw_data"

# Processed data output folder: power_dem_for_model/data/processed_data/
PROC_FOLDER = BASE_DIR / "data" / "processed_data"

# Raw electricity meter data file
RAW_FILE = RAW_FOLDER / "LD2011_2014.txt"

# ─────────────────────────────────────────────────────────────────────────────
# RESULTS DIRECTORIES
# ─────────────────────────────────────────────────────────────────────────────
# Root results folder: power_dem_for_model/results/
RESULTS_DIR = BASE_DIR / "results"

# Trained models storage: power_dem_for_model/results/Models/
MODELS_DIR = RESULTS_DIR / "Models"

# Model evaluation metrics: power_dem_for_model/results/model_results/
MODEL_RESULTS_DIR = RESULTS_DIR / "model_results"

# Forecast predictions & visualizations: power_dem_for_model/results/pred/
PRED_DIR = RESULTS_DIR / "pred"

# ─────────────────────────────────────────────────────────────────────────────
# AUTOMATIC DIRECTORY CREATION
# ─────────────────────────────────────────────────────────────────────────────
# Ensure all required folders exist at module import time
PROC_FOLDER.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)
MODEL_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
PRED_DIR.mkdir(parents=True, exist_ok=True)
