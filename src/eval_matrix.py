"""
================================================================================
RESULTS EXPORT MODULE
================================================================================

Purpose:
    Compile model evaluation metrics and export results to Excel with multiple
    sheets organized by strategy (ALL_METER, CATEGORY, INDIVIDUAL) and best
    models per forecasting horizon.

Key Functions:
    - save_results(): Export model performance metrics to Excel file

Author: Power Demand Forecasting Team
Version: 1.0
================================================================================
"""

from pathlib import Path
import pandas as pd

# ─── Configuration Import ────────────────────────────────────────────────────
from config.config import MODEL_RESULTS_DIR


def save_results(final_df: pd.DataFrame) -> Path:
    """
    Export model evaluation metrics to Excel workbook with strategy breakdowns.

    Creates an Excel file with 4 worksheets:
    1. ALL_METER: Aggregated demand models (SARIMAX + Prophet)
    2. CATEGORY: Season-category stratified models
    3. INDIVIDUAL: Individual meter + aggregated models
    4. BEST_BY_HORIZON: Top performing model per forecast horizon

    Parameters
    ----------
    final_df : pd.DataFrame
        Model results dataframe with columns:
        [Strategy, Model, Horizon, Params, Structure, sMAPE (%), R2]

    Returns
    -------
    Path
        Path to saved Excel file

    Notes
    -----
    - Creates MODEL_RESULTS_DIR if it doesn't exist
    - Automatically sorts models by sMAPE (%) ascending (lower is better)
    - Best models selected as first row per horizon after sorting
    """
    # Create output directory if needed
    MODEL_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Define output file path
    out_file = MODEL_RESULTS_DIR / "evaluation_metrics_testing.xlsx"

    # ─────────────────────────────────────────────────────────────────────────
    # Select Best Models Per Horizon (lowest sMAPE)
    # ─────────────────────────────────────────────────────────────────────────
    # Sort by sMAPE ascending and select first model per horizon
    best_models = (
        final_df
        .sort_values("sMAPE (%)")
        .groupby("Horizon")
        .first()
        .reset_index()
    )

    # ─────────────────────────────────────────────────────────────────────────
    # Write Results to Excel with Multiple Worksheets
    # ─────────────────────────────────────────────────────────────────────────
    # Create ExcelWriter context manager for multi-sheet output
    with pd.ExcelWriter(out_file, engine="xlsxwriter") as writer:
        # Sheet 1: ALL_METER strategy (aggregate demand forecasting)
        final_df.query("Strategy=='ALL_METER'").to_excel(
            writer, "ALL_METER", index=False
        )
        
        # Sheet 2: CATEGORY strategy (season-stratified forecasting)
        final_df.query("Strategy=='CATEGORY'").to_excel(
            writer, "CATEGORY", index=False
        )
        
        # Sheet 3: INDIVIDUAL strategy (per-meter forecasting)
        final_df.query("Strategy=='INDIVIDUAL'").to_excel(
            writer, "INDIVIDUAL", index=False
        )
        
        # Sheet 4: Best models ranked by horizon and sMAPE
        best_models.to_excel(
            writer, "BEST_BY_HORIZON", index=False
        )

    return out_file
