"""
Data Preprocessing Module for Retail Sales Forecasting.
Handles loading, cleaning, missing values, datetime parsing, and feature engineering.
"""

import os
import sys
import pandas as pd
import numpy as np
import requests

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config import (
    DATA_URL,
    RAW_CSV_PATH,
    PROCESSED_CSV_PATH,
    DAILY_AGGREGATE_PATH,
    DATE_COL,
    SALES_COL,
    CATEGORY_COL,
    QUANTITY_COL,
)


def download_data(url: str = DATA_URL, save_path: str = RAW_CSV_PATH) -> str:
    """
    Download retail sales dataset from URL if not already present.
    Returns path to the CSV file.
    """
    if os.path.exists(save_path):
        return save_path
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    with open(save_path, "wb") as f:
        f.write(response.content)
    return save_path


def load_raw_data(csv_path: str = RAW_CSV_PATH) -> pd.DataFrame:
    """Load raw CSV into a DataFrame."""
    if not os.path.exists(csv_path):
        download_data(save_path=csv_path)
    df = pd.read_csv(csv_path)
    return df


def parse_dates(df: pd.DataFrame, date_column: str = DATE_COL) -> pd.DataFrame:
    """
    Parse date column. Handles formats like '1/5/2019' (M/D/YYYY).
    """
    df = df.copy()
    df[date_column] = pd.to_datetime(df[date_column], format="mixed", dayfirst=False)
    return df


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Handle missing values: drop rows with critical missing, fill optional.
    """
    df = df.copy()
    # Critical columns: no missing allowed for analysis
    critical = [DATE_COL, SALES_COL]
    for col in critical:
        if col in df.columns and df[col].isna().any():
            df = df.dropna(subset=[col])
    # Fill numeric with median for optional columns
    numeric = df.select_dtypes(include=[np.number]).columns
    for col in numeric:
        if col in df.columns and df[col].isna().any():
            df[col] = df[col].fillna(df[col].median())
    # Fill categorical with mode
    cat = df.select_dtypes(include=["object"]).columns
    for col in cat:
        if col in df.columns and df[col].isna().any():
            df[col] = df[col].fillna(df[col].mode().iloc[0] if len(df[col].mode()) else "Unknown")
    return df


def add_time_features(df: pd.DataFrame, date_column: str = DATE_COL) -> pd.DataFrame:
    """
    Feature engineering: month, year, weekday, quarter, and seasonality.
    """
    df = df.copy()
    df["month"] = df[date_column].dt.month
    df["year"] = df[date_column].dt.year
    df["weekday"] = df[date_column].dt.weekday  # 0=Monday, 6=Sunday
    df["quarter"] = df[date_column].dt.quarter
    # Seasonality: 1=Winter, 2=Spring, 3=Summer, 4=Fall (Northern hemisphere)
    df["season"] = ((df["month"] % 12 + 3) // 3).map(
        {1: "Winter", 2: "Spring", 3: "Summer", 4: "Fall"}
    )
    df["month_name"] = df[date_column].dt.month_name()
    return df


def clean_sales_column(df: pd.DataFrame, sales_column: str = SALES_COL) -> pd.DataFrame:
    """Ensure sales column is numeric and non-negative."""
    df = df.copy()
    df[sales_column] = pd.to_numeric(df[sales_column], errors="coerce")
    df = df[df[sales_column] >= 0]
    return df


def preprocess_pipeline(
    csv_path: str = RAW_CSV_PATH,
    save_processed: bool = True,
    save_daily: bool = True,
) -> pd.DataFrame:
    """
    Full preprocessing pipeline: load -> parse dates -> missing values -> features -> clean.
    Optionally saves processed transaction-level and daily aggregate CSVs.
    """
    df = load_raw_data(csv_path)
    df = parse_dates(df)
    df = handle_missing_values(df)
    df = add_time_features(df)
    df = clean_sales_column(df)

    os.makedirs(os.path.dirname(PROCESSED_CSV_PATH), exist_ok=True)
    if save_processed:
        df.to_csv(PROCESSED_CSV_PATH, index=False)

    if save_daily:
        daily = aggregate_daily_sales(df)
        daily.to_csv(DAILY_AGGREGATE_PATH, index=False)

    return df


def aggregate_daily_sales(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate sales by date (and optionally by category) for time series modeling.
    """
    agg = (
        df.groupby(DATE_COL)
        .agg(
            sales=pd.NamedAgg(SALES_COL, "sum"),
            quantity=pd.NamedAgg(QUANTITY_COL, "sum"),
            transactions=pd.NamedAgg(SALES_COL, "count"),
        )
        .reset_index()
    )
    agg = add_time_features(agg, date_column=DATE_COL)
    return agg


def get_daily_series(df: pd.DataFrame) -> pd.DataFrame:
    """Return daily aggregated series from processed transaction data."""
    return aggregate_daily_sales(df)


def get_processed_data(force_reprocess: bool = False) -> pd.DataFrame:
    """
    Load processed data from disk or run preprocessing if missing/forced.
    """
    if force_reprocess or not os.path.exists(PROCESSED_CSV_PATH):
        download_data(save_path=RAW_CSV_PATH)
        return preprocess_pipeline()
    return pd.read_csv(PROCESSED_CSV_PATH, parse_dates=[DATE_COL])


def get_daily_data(force_reprocess: bool = False) -> pd.DataFrame:
    """Load daily aggregate for forecasting."""
    if force_reprocess or not os.path.exists(DAILY_AGGREGATE_PATH):
        get_processed_data(force_reprocess=True)
    return pd.read_csv(DAILY_AGGREGATE_PATH, parse_dates=[DATE_COL])


if __name__ == "__main__":
    # Run preprocessing when executed as script
    download_data()
    df = preprocess_pipeline()
    print("Processed shape:", df.shape)
    print("Date range:", df[DATE_COL].min(), "to", df[DATE_COL].max())
    print("Columns:", list(df.columns))
