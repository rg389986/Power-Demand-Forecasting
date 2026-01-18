"""
================================================================================
MODEL BUILDING MODULE
================================================================================

Purpose:
    Build and evaluate time-series forecasting models using SARIMAX and Prophet
    algorithms across three forecasting strategies (ALL_METER, CATEGORY, INDIVIDUAL).
    Supports multiple hyperparameter configurations and computes performance metrics.

Key Functions:
    - run_strategies(): Main orchestration function
    - run_model(): Train and forecast with SARIMAX or Prophet
    - evaluate(): Compute sMAPE and R² metrics
    - build_category_df(): Prepare data for category-based models

Author: Power Demand Forecasting Team
Version: 1.0
================================================================================
"""

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from prophet import Prophet
from sklearn.metrics import r2_score

# ─── Model Constants ─────────────────────────────────────────────────────────
EPSILON = 1e-6  # Small value to avoid division by zero in sMAPE

# SARIMAX hyperparameter grid: (p, d, q)
SARIMAX_ORDERS = [(1, 1, 1), (2, 1, 2), (2, 0, 2)]

# SARIMAX seasonal orders: (P, D, Q, s) where s=7 for weekly seasonality
SEASONAL_ORDERS = [(1, 1, 1, 7), (1, 0, 1, 7), (2, 1, 1, 7)]

# Prophet configuration variants (seasonality modes)
PROPHET_CONFIGS = [
    {"weekly_seasonality": True, "yearly_seasonality": True, "seasonality_mode": "additive"},
    {"weekly_seasonality": True, "yearly_seasonality": True, "seasonality_mode": "multiplicative"},
    {"weekly_seasonality": False, "yearly_seasonality": True, "seasonality_mode": "additive"},
]


# ─────────────────────────────────────────────────────────────────────────────
# HELPER: Build Category DataFrame
# ─────────────────────────────────────────────────────────────────────────────
def build_category_df(df_daily: pd.DataFrame, meter_list: list, season_cols: list) -> pd.DataFrame:
    """
    Create a category-level dataframe with aggregated load and exogenous features.

    Groups specified meters by date and appends seasonal/holiday features.

    Parameters
    ----------
    df_daily : pd.DataFrame
        Daily data with meter columns and features
    meter_list : list
        List of meter column names to aggregate
    season_cols : list
        Season feature column names (from feature engineering)

    Returns
    -------
    pd.DataFrame
        Merged dataframe with aggregated load and exogenous regressors
    """
    # Filter to only meters that exist in dataframe
    meter_cols = [m for m in meter_list if m in df_daily.columns]

    # Sum selected meters to create category-level load
    df = df_daily[["Year", "Month", "Day"] + meter_cols].copy()
    df["total_load"] = df[meter_cols].sum(axis=1)

    # Exogenous features for model input
    feature_cols = ["holiday_flag"] + season_cols

    # Merge features by date key (Year/Month/Day)
    return df.merge(
        df_daily[["Year", "Month", "Day"] + feature_cols].drop_duplicates(),
        on=["Year", "Month", "Day"],
        how="left"
    )


# ─────────────────────────────────────────────────────────────────────────────
# HELPER: Evaluate Model Performance
# ─────────────────────────────────────────────────────────────────────────────
def evaluate(actual: np.ndarray, pred: np.ndarray) -> tuple:
    """
    Compute symmetric Mean Absolute Percentage Error (sMAPE) and R² metrics.

    sMAPE = 200/n * Σ|actual - pred| / (|actual| + |pred|)

    Parameters
    ----------
    actual : np.ndarray
        Ground truth values
    pred : np.ndarray
        Model predictions

    Returns
    -------
    tuple
        (smape: float, r2: float) or (np.nan, np.nan) if insufficient valid data

    Notes
    -----
    - Clips values to EPSILON to avoid division by zero
    - Returns NaN if fewer than 2 valid finite values
    - sMAPE scale: 0% = perfect, 100% = random baseline
    """
    # Convert to numeric arrays
    actual = np.asarray(actual, float)
    pred = np.asarray(pred, float)

    # Create mask for finite values (remove NaN/inf)
    mask = np.isfinite(actual) & np.isfinite(pred)
    if mask.sum() < 2:
        return np.nan, np.nan

    # Extract finite values and clip to positive epsilon
    actual = np.clip(actual[mask], EPSILON, None)
    pred = np.clip(pred[mask], EPSILON, None)

    # Compute symmetric MAPE
    smape = (2 * np.abs(pred - actual) /
             (np.abs(actual) + np.abs(pred))).mean() * 100

    # Compute R² score
    return smape, r2_score(actual, pred)


# ─────────────────────────────────────────────────────────────────────────────
# CORE: Model Training & Forecasting
# ─────────────────────────────────────────────────────────────────────────────
def run_model(df: pd.DataFrame, season_cols: list, model_type: str, cfg: dict, horizon: int) -> np.ndarray:
    """
    Train a forecasting model and generate predictions for specified horizon.

    Supports two model types:
    1. SARIMAX: Seasonal ARIMA with exogenous regressors
    2. PROPHET: Facebook's time-series model with custom regressors

    Parameters
    ----------
    df : pd.DataFrame
        Training dataframe with columns: Year, Month, Day, total_load, holiday_flag, season columns
    season_cols : list
        Season feature column names
    model_type : str
        Either 'SARIMAX' or 'PROPHET'
    cfg : dict
        Model configuration (e.g., {'order': (1,1,1), 'seasonal': (1,1,1,7)} for SARIMAX)
    horizon : int
        Number of days ahead to forecast

    Returns
    -------
    np.ndarray
        Forecast predictions for 'horizon' days
    """
    df = df.copy()

    # Create datetime column from Year/Month/Day
    df["ds"] = pd.to_datetime(
        dict(year=df.Year, month=df.Month, day=df.Day)
    )

    # Prepare target and exogenous variables
    y = df["total_load"].astype(float)
    exog = df[["holiday_flag"] + season_cols].fillna(0)

    if model_type == "SARIMAX":
        # ─── SARIMAX Model ──────────────────────────────────────────────────
        model = SARIMAX(
            y,
            exog=exog,
            order=cfg["order"],
            seasonal_order=cfg["seasonal"],
            enforce_stationarity=False,  # Allow non-stationary data
            enforce_invertibility=False
        ).fit(disp=False)

        # Forecast with exogenous data for specified horizon
        fc = model.get_forecast(steps=horizon, exog=exog.iloc[-horizon:])
        return fc.predicted_mean.values

    else:  # PROPHET
        # ─── PROPHET Model ──────────────────────────────────────────────────
        df_p = df[["ds", "total_load", "holiday_flag"] + season_cols].copy()
        df_p["y"] = df_p["total_load"]
        df_p.drop(columns="total_load", inplace=True)

        # Initialize and configure Prophet model
        m = Prophet(**cfg)
        m.add_regressor("holiday_flag")
        for c in season_cols:
            m.add_regressor(c)

        # Fit on training data
        m.fit(df_p)
        
        # Generate forecast for last 'horizon' dates with their features
        future = df_p.iloc[-horizon:]
        return m.predict(future)["yhat"].values


# ─────────────────────────────────────────────────────────────────────────────
# ORCHESTRATION: Run All Forecasting Strategies
# ─────────────────────────────────────────────────────────────────────────────
def run_strategies(df_daily: pd.DataFrame, meter_summary_df: pd.DataFrame, season_cols: list, horizons: list) -> pd.DataFrame:
    """
    Train models across three forecasting strategies and multiple horizons.

    Strategies:
    1. ALL_METER: Single model on total aggregated demand
    2. CATEGORY: Separate models for normal vs outlier meters (by 99th percentile)
    3. INDIVIDUAL: Sum of normal meters + individual models for outliers

    Parameters
    ----------
    df_daily : pd.DataFrame
        Daily data with features from build_time_features()
    meter_summary_df : pd.DataFrame
        Meter statistics (from data_cleaning)
    season_cols : list
        Season feature column names
    horizons : list
        List of forecast horizons in days (e.g., [30, 60, 90])

    Returns
    -------
    pd.DataFrame
        Results dataframe with columns:
        [Strategy, Model, Horizon, Params, Structure, sMAPE (%), R2]
    """
    # ─────────────────────────────────────────────────────────────────────────
    # STEP 1: Categorize Meters by Load Profile (Normal vs Outlier)
    # ─────────────────────────────────────────────────────────────────────────
    # Use 99th percentile of daily average load as threshold
    threshold = meter_summary_df["Per_Day_Avg_Load"].quantile(0.99)

    meter_summary_df["meter_category"] = np.where(
        meter_summary_df["Per_Day_Avg_Load"] > threshold,
        "outlier_meter",
        "normal_meter"
    )

    # Create meter-to-category mapping
    cat_meter_map = (
        meter_summary_df
        .groupby("meter_category")["Meter"]
        .apply(list)
        .to_dict()
    )

    # Compute ground truth: sum of all meters for evaluation
    total_actual = df_daily[
        cat_meter_map["normal_meter"] + cat_meter_map["outlier_meter"]
    ].sum(axis=1)

    results = []

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 2: Model Training Loop Over Horizons
    # ─────────────────────────────────────────────────────────────────────────
    for horizon in horizons:
        # Extract test set (last 'horizon' days)
        actual = total_actual.iloc[-horizon:].values

        # ┌─ STRATEGY 1: ALL_METER ─────────────────────────────────────────────
        df_all = build_category_df(
            df_daily,
            cat_meter_map["normal_meter"] + cat_meter_map["outlier_meter"],
            season_cols
        )

        # Test all SARIMAX order combinations
        for o in SARIMAX_ORDERS:
            for so in SEASONAL_ORDERS:
                pred = run_model(
                    df_all, season_cols, "SARIMAX",
                    {"order": o, "seasonal": so}, horizon
                )
                sm, r2 = evaluate(actual, pred)
                results.append(["ALL_METER", "SARIMAX", horizon, str(o), str(so), sm, r2])

        # Test Prophet configurations
        for cfg in PROPHET_CONFIGS:
            pred = run_model(df_all, season_cols, "PROPHET", cfg, horizon)
            sm, r2 = evaluate(actual, pred)
            results.append(["ALL_METER", "PROPHET", horizon, str(cfg), None, sm, r2])

        # ┌─ STRATEGY 2: CATEGORY ──────────────────────────────────────────────
        # Build separate dataframes for normal and outlier meters
        df_n = build_category_df(df_daily, cat_meter_map["normal_meter"], season_cols)
        df_o = build_category_df(df_daily, cat_meter_map["outlier_meter"], season_cols)

        # Test Prophet on combined category forecasts
        for cfg in PROPHET_CONFIGS:
            pred = (
                run_model(df_n, season_cols, "PROPHET", cfg, horizon) +
                run_model(df_o, season_cols, "PROPHET", cfg, horizon)
            )
            sm, r2 = evaluate(actual, pred)
            results.append(["CATEGORY", "PROPHET", horizon, str(cfg), "N+O", sm, r2])

        # ┌─ STRATEGY 3: INDIVIDUAL ────────────────────────────────────────────
        # Aggregate normal meters + individual models for each outlier meter
        pred_total = run_model(df_n, season_cols, "PROPHET", PROPHET_CONFIGS[0], horizon)

        for mtr in cat_meter_map["outlier_meter"]:
            # Create meter-level dataframe
            df_m = df_daily[["Year", "Month", "Day", mtr]].rename(columns={mtr: "total_load"})
            df_m = df_m.merge(
                df_daily[["Year", "Month", "Day", "holiday_flag"] + season_cols],
                on=["Year", "Month", "Day"], how="left"
            )
            pred_total += run_model(df_m, season_cols, "PROPHET", PROPHET_CONFIGS[0], horizon)

        sm, r2 = evaluate(actual, pred_total)
        results.append(["INDIVIDUAL", "PROPHET", horizon, "NORMAL+INDIV", None, sm, r2])

    # ─────────────────────────────────────────────────────────────────────────
    # Compile Results into DataFrame
    # ─────────────────────────────────────────────────────────────────────────
    return pd.DataFrame(
        results,
        columns=["Strategy", "Model", "Horizon", "Params", "Structure", "sMAPE (%)", "R2"]
    )
