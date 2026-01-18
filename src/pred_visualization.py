# ==================================================
# PREDICTION + VISUALIZATION (REQUESTED DAYS ONLY)
# ==================================================

import ast
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from pathlib import Path
from statsmodels.tsa.statespace.sarimax import SARIMAX
from prophet import Prophet

# --------------------------------------------------
# IMPORT PATHS FROM CONFIG (ONLY CHANGE)
# --------------------------------------------------
from config.config import (
    MODELS_DIR,
    MODEL_RESULTS_DIR,
    PRED_DIR
)


# --------------------------------------------------
# SELECT MODEL HORIZON
# --------------------------------------------------
def select_model_horizon(requested, available):
    available = sorted(available)

    if requested in available:
        return requested

    higher = [h for h in available if h > requested]
    if higher:
        return min(higher)

    return max(available)


# --------------------------------------------------
# MAIN FUNCTION
# --------------------------------------------------
def run_prediction_visualization(
    df_daily: pd.DataFrame,
    season_cols: list,
    requested_horizon: int
):

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 1: Load Best Model Configuration
    # ─────────────────────────────────────────────────────────────────────────
    # Read model evaluation results to get best-performing configuration
    best_by_horizon = pd.read_excel(
        MODEL_RESULTS_DIR / "evaluation_metrics_testing.xlsx",
        sheet_name="BEST_BY_HORIZON"
    )

    available_horizons = best_by_horizon["Horizon"].unique().tolist()
    model_horizon = select_model_horizon(
        requested_horizon, available_horizons
    )

    # Extract best model config for selected horizon
    best_cfg = best_by_horizon.loc[
        best_by_horizon["Horizon"] == model_horizon
    ].iloc[0]

    strategy = best_cfg["Strategy"]
    model_type = best_cfg["Model"]
    params = best_cfg["Params"]
    structure = best_cfg["Structure"]

    print("\n🎯 Prediction Configuration")
    print(f"   Requested Horizon : {requested_horizon} days")
    print(f"   Model Horizon     : {model_horizon} days")
    print(f"   Strategy          : {strategy}")
    print(f"   Model Type        : {model_type}")

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 2: Prepare Data
    # ─────────────────────────────────────────────────────────────────────────
    # Convert date components to datetime index for forecasting
    df = df_daily.copy()
    df["ds"] = pd.to_datetime(
        dict(year=df.Year, month=df.Month, day=df.Day)
    )

    ts = df["total_load"].astype(float)
    exog = df[["holiday_flag"] + season_cols].fillna(0)

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 3: Define Model Save Path
    # ─────────────────────────────────────────────────────────────────────────
    MODEL_PATH = MODELS_DIR / (
        f"final_{strategy.lower()}_{model_type.lower()}_{model_horizon}d.joblib"
    )

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 4: Train and Save Model
    # ─────────────────────────────────────────────────────────────────────────
    if model_type == "SARIMAX":
        # ─── SARIMAX Model Training ──────────────────────────────────────────
        order = ast.literal_eval(params)
        seasonal = ast.literal_eval(structure)

        model = SARIMAX(
            ts[:-model_horizon],
            exog=exog.iloc[:-model_horizon],
            order=order,
            seasonal_order=seasonal,
            enforce_stationarity=False,
            enforce_invertibility=False
        ).fit(disp=False)

        # Serialize model to disk
        joblib.dump(model, MODEL_PATH)

        # Generate predictions for evaluation window
        fc_eval = model.get_forecast(
            steps=model_horizon,
            exog=exog.iloc[-model_horizon:]
        )
        pred_eval_full = fc_eval.predicted_mean.values

    else:  # PROPHET
        # ─── PROPHET Model Training ──────────────────────────────────────────
        cfg = ast.literal_eval(params)

        df_p = df[["ds", "total_load", "holiday_flag"] + season_cols].copy()
        df_p["y"] = df_p["total_load"]
        df_p.drop(columns="total_load", inplace=True)

        model = Prophet(**cfg)
        model.add_regressor("holiday_flag")
        for c in season_cols:
            model.add_regressor(c)

        # Fit on all data except evaluation window
        model.fit(df_p.iloc[:-model_horizon])
        
        # Serialize model to disk
        joblib.dump(model, MODEL_PATH)

        # Generate predictions for evaluation window
        pred_eval_full = model.predict(
            df_p.iloc[-model_horizon:]
        )["yhat"].values

    print(f"✅ Model trained and saved: {MODEL_PATH}")

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 5: Evaluation on Requested Horizon Window
    # ─────────────────────────────────────────────────────────────────────────
    # Extract actual values and predictions for requested horizon
    actual = ts.iloc[-requested_horizon:].values
    pred_eval = pred_eval_full[-requested_horizon:]
    dates_eval = df["ds"].iloc[-requested_horizon:]

    # Compute residuals (errors)
    residuals = actual - pred_eval

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 6: Plot 1 - Actual vs Predicted (Evaluation Period)
    # ─────────────────────────────────────────────────────────────────────────
    plt.figure(figsize=(12, 6))
    plt.plot(dates_eval, actual, label="Actual", marker="o", linewidth=2)
    plt.plot(dates_eval, pred_eval, label="Predicted", marker="s", linewidth=2, linestyle="--")
    plt.title(
        f"Load Forecast Evaluation ({requested_horizon} Days | "
        f"Model Trained on {model_horizon}-day Horizon)",
        fontsize=14, fontweight='bold'
    )
    plt.xlabel("Date", fontsize=12)
    plt.ylabel("Total Load (kWh)", fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 7: Plot 2 - Residual Analysis
    # ─────────────────────────────────────────────────────────────────────────
    plt.figure(figsize=(12, 4))
    plt.plot(dates_eval, residuals, marker="o", linewidth=2, color="orange")
    plt.axhline(0, linestyle="--", color="red", alpha=0.7, label="Zero Error")
    plt.title("Forecast Residuals (Actual − Predicted)", fontsize=14, fontweight='bold')
    plt.xlabel("Date", fontsize=12)
    plt.ylabel("Residual Load (kWh)", fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 8: Future Forecast (Beyond Training Data)
    # ─────────────────────────────────────────────────────────────────────────
    # Create future date range starting from day after last training date
    future_dates = pd.date_range(
        start=df["ds"].iloc[-1] + pd.Timedelta(days=1),
        periods=requested_horizon,
        freq="D"
    )

    if model_type == "SARIMAX":
        # ─── SARIMAX Future Forecast ────────────────────────────────────────
        future_exog = exog.iloc[-model_horizon:].iloc[:requested_horizon].copy()
        future_exog.index = future_dates

        fc_future = model.get_forecast(
            steps=requested_horizon,
            exog=future_exog
        )
        future_pred = fc_future.predicted_mean.values

    else:
        # ─── PROPHET Future Forecast ────────────────────────────────────────
        future_df = pd.DataFrame({"ds": future_dates})
        
        # Use last available values for exogenous features
        for c in ["holiday_flag"] + season_cols:
            future_df[c] = df_p[c].iloc[-1]

        future_pred = model.predict(future_df)["yhat"].values

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 9: Export Future Predictions to Excel
    # ─────────────────────────────────────────────────────────────────────────
    future_out = pd.DataFrame({
        "date": future_dates,
        "predicted_load": future_pred,
        "requested_horizon": requested_horizon,
        "model_horizon_used": model_horizon
    })

    out_path = PRED_DIR / f"future_prediction_{requested_horizon}d.xlsx"
    future_out.to_excel(out_path, index=False)

    print(f"📊 Future forecast exported: {out_path}")

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 10: Plot 3 - Future Forecast
    # ─────────────────────────────────────────────────────────────────────────
    plt.figure(figsize=(12, 6))
    plt.plot(future_dates, future_pred, marker="s", linewidth=2, label="Future Forecast", color="green")
    plt.title(f"Next {requested_horizon}-Day Load Forecast", fontsize=14, fontweight='bold')
    plt.xlabel("Date", fontsize=12)
    plt.ylabel("Total Load (kWh)", fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
