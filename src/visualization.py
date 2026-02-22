"""
Visualization Module for Retail Sales Forecasting.
EDA plots: sales trends, seasonal patterns, category revenue, monthly growth, distributions.
"""

import os
import sys
from typing import Optional

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config import DATE_COL, SALES_COL, CATEGORY_COL

# Style
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (10, 5)


def plot_sales_trend(
    df: pd.DataFrame,
    date_col: str = DATE_COL,
    sales_col: str = SALES_COL,
    agg: str = "D",
    ax=None,
) -> plt.Axes:
    """Plot daily or monthly sales trend over time."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 4))
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    trend = df.set_index(date_col)[sales_col].resample(agg).sum()
    trend.plot(ax=ax, color="steelblue", linewidth=1.5)
    ax.set_title("Sales Trend Over Time")
    ax.set_xlabel("Date")
    ax.set_ylabel("Total Sales")
    ax.legend(["Sales"])
    return ax


def plot_monthly_growth(
    df: pd.DataFrame,
    date_col: str = DATE_COL,
    sales_col: str = SALES_COL,
    ax=None,
) -> plt.Axes:
    """Plot monthly sales with month-over-month growth indication."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 4))
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    monthly = df.set_index(date_col)[sales_col].resample("ME").sum()
    monthly_pct = monthly.pct_change() * 100
    x = range(len(monthly))
    colors = ["green" if v >= 0 else "red" for v in monthly_pct.fillna(0)]
    ax.bar(x, monthly.values, color=colors, alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels([d.strftime("%Y-%m") for d in monthly.index], rotation=45)
    ax.set_title("Monthly Sales (Green = Growth, Red = Decline)")
    ax.set_xlabel("Month")
    ax.set_ylabel("Sales")
    return ax


def plot_category_revenue(
    df: pd.DataFrame,
    category_col: str = CATEGORY_COL,
    sales_col: str = SALES_COL,
    top_n: int = 10,
    ax=None,
) -> plt.Axes:
    """Bar chart of revenue by product category."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 5))
    rev = df.groupby(category_col)[sales_col].sum().sort_values(ascending=False).head(top_n)
    rev.plot(kind="barh", ax=ax, color="teal", alpha=0.8)
    ax.set_title("Revenue by Product Category")
    ax.set_xlabel("Total Revenue")
    ax.set_ylabel("Category")
    return ax


def plot_revenue_distribution(
    df: pd.DataFrame,
    sales_col: str = SALES_COL,
    ax=None,
) -> plt.Axes:
    """Distribution of transaction-level revenue."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4))
    df[sales_col].hist(ax=ax, bins=50, color="coral", alpha=0.7, edgecolor="white")
    ax.set_title("Revenue Distribution (per transaction)")
    ax.set_xlabel("Revenue")
    ax.set_ylabel("Frequency")
    return ax


def plot_seasonality(
    df: pd.DataFrame,
    date_col: str = DATE_COL,
    sales_col: str = SALES_COL,
    ax=None,
) -> plt.Axes:
    """Average sales by month (seasonal pattern)."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 4))
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df["month"] = df[date_col].dt.month
    month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    by_month = df.groupby("month")[sales_col].mean()
    by_month = by_month.reindex(range(1, 13)).fillna(0)  # ensure all 12 months for consistent x-axis
    by_month.plot(kind="bar", ax=ax, color="mediumpurple", alpha=0.8)
    ax.set_xticklabels(month_names, rotation=0)
    ax.set_title("Average Sales by Month (Seasonality)")
    ax.set_xlabel("Month")
    ax.set_ylabel("Average Sales")
    return ax


def plot_weekday_pattern(
    df: pd.DataFrame,
    date_col: str = DATE_COL,
    sales_col: str = SALES_COL,
    ax=None,
) -> plt.Axes:
    """Sales by weekday."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4))
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df["weekday"] = df[date_col].dt.weekday
    by_wd = df.groupby("weekday")[sales_col].sum()
    days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    by_wd = by_wd.reindex(range(7)).fillna(0)
    by_wd.index = days
    by_wd.plot(kind="bar", ax=ax, color="steelblue", alpha=0.8)
    ax.set_title("Sales by Weekday")
    ax.set_xlabel("Weekday")
    ax.set_ylabel("Total Sales")
    return ax


def plot_forecast_vs_actual(
    actual_dates: pd.Series,
    actual_values: pd.Series,
    forecast_dates: pd.Series,
    forecast_values: pd.Series,
    ax=None,
) -> plt.Axes:
    """Overlay actual and forecasted sales."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(actual_dates, actual_values, label="Actual", color="steelblue", linewidth=2)
    ax.plot(forecast_dates, forecast_values, label="Forecast", color="orange", linestyle="--", linewidth=1.5)
    ax.set_title("Sales: Actual vs Forecast")
    ax.set_xlabel("Date")
    ax.set_ylabel("Sales")
    ax.legend()
    return ax


def create_eda_figures(
    df: pd.DataFrame,
    output_dir: Optional[str] = None,
) -> list:
    """
    Generate all EDA figures. If output_dir is set, save figures there.
    Returns list of (name, fig) for in-memory use.
    """
    figures = []
    # 1. Sales trend
    fig, ax = plt.subplots(figsize=(12, 4))
    plot_sales_trend(df, ax=ax)
    fig.tight_layout()
    figures.append(("sales_trend", fig))
    if output_dir:
        fig.savefig(os.path.join(output_dir, "sales_trend.png"), dpi=100, bbox_inches="tight")
    plt.close(fig)

    # 2. Monthly growth
    fig, ax = plt.subplots(figsize=(10, 4))
    plot_monthly_growth(df, ax=ax)
    fig.tight_layout()
    figures.append(("monthly_growth", fig))
    if output_dir:
        fig.savefig(os.path.join(output_dir, "monthly_growth.png"), dpi=100, bbox_inches="tight")
    plt.close(fig)

    # 3. Category revenue
    fig, ax = plt.subplots(figsize=(10, 5))
    plot_category_revenue(df, ax=ax)
    fig.tight_layout()
    figures.append(("category_revenue", fig))
    if output_dir:
        fig.savefig(os.path.join(output_dir, "category_revenue.png"), dpi=100, bbox_inches="tight")
    plt.close(fig)

    # 4. Revenue distribution
    fig, ax = plt.subplots(figsize=(8, 4))
    plot_revenue_distribution(df, ax=ax)
    fig.tight_layout()
    figures.append(("revenue_distribution", fig))
    if output_dir:
        fig.savefig(os.path.join(output_dir, "revenue_distribution.png"), dpi=100, bbox_inches="tight")
    plt.close(fig)

    # 5. Seasonality
    fig, ax = plt.subplots(figsize=(10, 4))
    plot_seasonality(df, ax=ax)
    fig.tight_layout()
    figures.append(("seasonality", fig))
    if output_dir:
        fig.savefig(os.path.join(output_dir, "seasonality.png"), dpi=100, bbox_inches="tight")
    plt.close(fig)

    # 6. Weekday pattern
    fig, ax = plt.subplots(figsize=(8, 4))
    plot_weekday_pattern(df, ax=ax)
    fig.tight_layout()
    figures.append(("weekday_pattern", fig))
    if output_dir:
        fig.savefig(os.path.join(output_dir, "weekday_pattern.png"), dpi=100, bbox_inches="tight")
    plt.close(fig)

    return figures


if __name__ == "__main__":
    from src.data_preprocessing import get_processed_data
    from config import RESULTS_DIR
    df = get_processed_data()
    create_eda_figures(df, output_dir=RESULTS_DIR)
    print("EDA figures saved to", RESULTS_DIR)
