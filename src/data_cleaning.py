"""
================================================================================
DATA CLEANING MODULE
================================================================================

Purpose:
    Load, clean, and preprocess raw electricity meter data from the LD2011_2014
    dataset. Performs decimal correction, daily aggregation, feature extraction,
    and generates summary statistics for meter-level analysis.

Key Functions:
    - run_data_cleaning(): Main entry point for data pipeline

Author: Power Demand Forecasting Team
Version: 1.0
================================================================================
"""

import pandas as pd
from pathlib import Path


def run_data_cleaning(raw_file: Path, proc_folder: Path):
    """
    Execute complete data cleaning and preprocessing pipeline.

    The function performs the following operations:
    1. Load raw CSV with semicolon delimiter
    2. Convert comma decimals to dots (European format fix)
    3. Aggregate hourly data to daily totals
    4. Extract temporal features (Year, Month, Day, DayOfWeek, etc.)
    5. Identify problematic meters (excessive zeros, activation patterns)
    6. Generate meter-level summary statistics
    7. Save all outputs to CSV files

    Parameters
    ----------
    raw_file : Path
        Path to raw data file (e.g., LD2011_2014.txt)
    proc_folder : Path
        Output folder for processed data CSV files

    Returns
    -------
    tuple
        (df_hourly, df_daily, problematic_df, meter_summary_df)
        - df_hourly (pd.DataFrame): Hourly data with total_load column
        - df_daily (pd.DataFrame): Daily aggregated data with temporal features
        - problematic_df (pd.DataFrame): Meters with zero-count issues
        - meter_summary_df (pd.DataFrame): Per-meter activation & load stats

    Notes
    -----
    - Filters out year 2015 data (incomplete year)
    - Decimal format fix: comma → dot conversion for numeric columns
    - Daily aggregation uses .sum() (additive)
    """

    # ==================================================
    # 1. Read raw data
    # ==================================================
    df = pd.read_csv(raw_file, sep=';', quotechar='"')

    df.rename(columns={df.columns[0]: "timestamp"}, inplace=True)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.set_index("timestamp", inplace=True)

    # ==================================================
    # 2. Clean numeric columns + total load
    # ==================================================
    num_cols = df.columns

    df[num_cols] = df[num_cols].apply(
        lambda c: c.astype(str).str.replace(',', '.', regex=False)
    )
    df[num_cols] = df[num_cols].astype(float)

    df["total_load"] = df.sum(axis=1)
    df_hourly = df.copy()

    # ==================================================
    # 3. Daily aggregation + features
    # ==================================================
    df_daily = df.resample("D").sum()

    meter_cols = df_daily.columns.difference(["total_load"])
    df_daily["Zero_count"] = (df_daily[meter_cols] == 0).sum(axis=1)

    df_daily["Year"] = df_daily.index.year
    df_daily["Month"] = df_daily.index.month
    df_daily["Day"] = df_daily.index.day
    df_daily["DayOfWeek"] = df_daily.index.dayofweek
    df_daily["DayName"] = df_daily.index.day_name()
    df_daily["WeekOfYear"] = df_daily.index.isocalendar().week
    df_daily["IsWeekend"] = df_daily["DayOfWeek"].isin([5, 6]).astype(int)

    df_daily = df_daily.reset_index().rename(columns={"timestamp": "Date"})
    df_daily = df_daily[df_daily["Year"] != 2015]

    # ==================================================
    # 4. Problematic meter identification
    # ==================================================
    mt_cols = [
        c for c in df_daily.select_dtypes(include="number").columns
        if c.startswith("MT")
    ]

    records = []

    for c in mt_cols:
        s = df_daily[["Date", c]].dropna()

        activated = s[s[c] > 0]
        if activated.empty:
            continue

        activation_date = activated["Date"].iloc[0]
        s_after = s[s["Date"] >= activation_date]

        zero_cnt = (s_after[c] == 0).sum()
        total_cnt = s_after[c].count()

        if zero_cnt > 0:
            records.append({
                "column_name": c,
                "first_positive_date": activation_date,
                "zero_count_after_activation": int(zero_cnt),
                "total_count_after_activation": int(total_cnt),
                "zero_percentage_after_activation": round(
                    (zero_cnt / total_cnt) * 100, 2
                )
            })

    problematic_df = pd.DataFrame(records)

    # ==================================================
    # 5. Meter-level summary
    # ==================================================
    records = []

    for c in mt_cols:
        s = df_daily[["Date", c]].dropna()
        activated = s[s[c] > 0]

        if activated.empty:
            continue

        activation_date = activated["Date"].iloc[0]
        s_after = s[s["Date"] >= activation_date]

        total_load = s_after[c].sum()
        active_days = len(s_after)

        monthly_sum = (
            s_after
            .groupby(s_after["Date"].dt.to_period("M"))[c]
            .sum()
        )

        records.append({
            "Meter": c,
            "Activation_Date": activation_date,
            "Total_Load": total_load,
            "Active_Days": active_days,
            "Per_Day_Avg_Load": round(total_load / active_days, 2),
            "Monthly_Avg_Load": round(monthly_sum.mean(), 2),
            "Active_Month_Count": monthly_sum.size
        })

    meter_summary_df = pd.DataFrame(records)

    # ==================================================
    # 6. Save outputs
    # ==================================================
    df_hourly.to_csv(proc_folder / "df_hourly.csv")
    df_daily.to_csv(proc_folder / "df_daily.csv", index=False)
    problematic_df.to_csv(proc_folder / "problematic_df.csv", index=False)
    meter_summary_df.to_csv(proc_folder / "meter_summary_df.csv", index=False)

    return df_hourly, df_daily, problematic_df, meter_summary_df
