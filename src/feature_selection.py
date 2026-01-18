"""
================================================================================
FEATURE ENGINEERING MODULE
================================================================================

Purpose:
    Extract temporal and categorical features from daily electricity load data.
    Creates season categories, holiday flags, and one-hot encoded features for
    use in time-series forecasting models.

Key Functions:
    - build_time_features(): Main feature engineering pipeline

Author: Power Demand Forecasting Team
Version: 1.0
================================================================================
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from pathlib import Path


def build_time_features(
    df_daily: pd.DataFrame,
    save: bool = False,
    proc_folder: Path | None = None
):
    """
    Extract and engineer temporal features for forecasting models.

    Operations performed:
    1. Classify months into season quartiles (winter/normal/summer/peak_summer)
    2. Create holiday flag for major holidays (Christmas, New Year)
    3. Create shutdown flag for day-before holidays (demand reduction period)
    4. One-hot encode season categories (drop first to avoid collinearity)

    Parameters
    ----------
    df_daily : pd.DataFrame
        Daily aggregated dataframe with columns:
        [Date, total_load, Year, Month, Day, DayOfWeek, DayName, WeekOfYear, IsWeekend, ...]
    save : bool, optional
        If True, persist enhanced dataframe to CSV (default: False)
    proc_folder : Path, optional
        Output folder for saving (required if save=True)

    Returns
    -------
    tuple
        (df_daily_enhanced, season_cols, month_avg)
        - df_daily_enhanced (pd.DataFrame): Input dataframe with new features
        - season_cols (list): One-hot encoded season column names
        - month_avg (pd.DataFrame): Monthly load with season mapping

    Raises
    ------
    ValueError
        If save=True but proc_folder is None

    Notes
    -----
    - Season categories use quartile-based binning on monthly average load
    - One-hot encoding uses drop='first' for regression models
    - Holiday flags: 12/25 (Christmas) and 1/1 (New Year)
    """

    df_daily = df_daily.copy()

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 1: Monthly Load Aggregation for Season Categorization
    # ─────────────────────────────────────────────────────────────────────────
    # Group by month and compute average daily load for seasonal patterns
    month_avg = (
        df_daily
        .groupby("Month", as_index=False)["total_load"]
        .mean()
        .sort_values("total_load")
    )

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 2: Assign Season Categories Based on Load Quartiles
    # ─────────────────────────────────────────────────────────────────────────
    # Create 4 season bins: winter (low), normal, summer, peak_summer (high)
    month_avg["season_category"] = pd.qcut(
        month_avg["total_load"],
        q=4,
        labels=["winter", "normal", "summer", "peak_summer"]
    )

    # Map season categories back to daily data by month
    season_map = dict(
        zip(month_avg["Month"], month_avg["season_category"])
    )
    df_daily["season_category"] = df_daily["Month"].map(season_map)

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 3: Holiday Flag (Major Holidays)
    # ─────────────────────────────────────────────────────────────────────────
    # Flag for Christmas (12/25) and New Year (1/1) - peak holidays with low demand
    df_daily["holiday_flag"] = (
        ((df_daily["Month"] == 12) & (df_daily["Day"] == 25)) |
        ((df_daily["Month"] == 1) & (df_daily["Day"] == 1))
    ).astype(int)

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 4: Shutdown Flag (Day Before Major Holidays)
    # ─────────────────────────────────────────────────────────────────────────
    # Demand reduction period the day before major holidays
    df_daily["Date"] = pd.to_datetime(df_daily["Date"])

    shutdown_dates = (
        df_daily.loc[df_daily["holiday_flag"] == 1, "Date"]
        - pd.Timedelta(days=1)
    )

    df_daily["shutdown_flag"] = df_daily["Date"].isin(shutdown_dates).astype(int)

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 5: One-Hot Encode Season Categories
    # ─────────────────────────────────────────────────────────────────────────
    # Create binary indicator columns for each season (drop first to avoid multicollinearity)
    ohe = OneHotEncoder(
        drop="first",  # Remove first category to avoid collinearity
        sparse_output=False,
        handle_unknown="ignore"
    )

    season_encoded = ohe.fit_transform(
        df_daily[["season_category"]]
    )

    # Extract feature names (season_category_<season_name>)
    season_cols = ohe.get_feature_names_out(
        ["season_category"]
    ).tolist()

    # Create dataframe with one-hot encoded columns
    season_dummies = pd.DataFrame(
        season_encoded,
        columns=season_cols,
        index=df_daily.index
    )

    # Concatenate new features to original dataframe
    df_daily = pd.concat([df_daily, season_dummies], axis=1)

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 6: Optional Data Persistence
    # ─────────────────────────────────────────────────────────────────────────
    if save:
        if proc_folder is None:
            raise ValueError("proc_folder must be provided when save=True")

        out_path = proc_folder / "df_daily_features.csv"
        df_daily.to_csv(out_path, index=False)

    return df_daily, season_cols, month_avg
