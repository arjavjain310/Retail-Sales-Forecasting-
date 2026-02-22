"""
Model Training Module for Retail Sales Forecasting.
Implements ARIMA, Prophet, and Linear Regression; evaluates with MAE and RMSE.
"""

import os
import sys
import json
import pickle
import warnings
from datetime import datetime

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Add project root
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config import (
    PROCESSED_DATA_PATH,
    MODELS_DIR,
    RESULTS_DIR,
    FORECAST_HORIZON_MONTHS,
    TEST_SIZE_MONTHS,
    PROPHET_MODEL_PATH,
    ARIMA_MODEL_PATH,
    LR_MODEL_PATH,
    FORECAST_CSV_PATH,
)
from src.data_preprocessing import get_daily_data, get_processed_data

warnings.filterwarnings("ignore", category=UserWarning)

# Optional: Prophet and ARIMA (may not be installed)
try:
    from prophet import Prophet
    HAS_PROPHET = True
except ImportError:
    HAS_PROPHET = False

try:
    from statsmodels.tsa.arima.model import ARIMA
    HAS_ARIMA = True
except ImportError:
    HAS_ARIMA = False


# -----------------------------------------------------------------------------
# Evaluation metrics
# -----------------------------------------------------------------------------
def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Compute MAE and RMSE."""
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
    }


# -----------------------------------------------------------------------------
# Train / predict helpers
# -----------------------------------------------------------------------------
def _get_series_and_cutoff(horizon_months: int = FORECAST_HORIZON_MONTHS):
    """Load daily series and split for train/test and forecast."""
    daily = get_daily_data()
    daily = daily.sort_values("Date").reset_index(drop=True)
    daily["ds"] = daily["Date"]
    daily["y"] = daily["sales"]

    # Use last 20% of days for test if series is short (< 4 months); else last TEST_SIZE_MONTHS
    n_days = (daily["Date"].max() - daily["Date"].min()).days
    if n_days < 120:  # ~4 months
        test_fraction = 0.2
        n_test = max(1, int(len(daily) * test_fraction))
        train = daily.iloc[:-n_test].copy()
        test = daily.iloc[-n_test:].copy()
    else:
        cutoff = daily["Date"].max() - pd.DateOffset(months=TEST_SIZE_MONTHS)
        train = daily[daily["Date"] <= cutoff].copy()
        test = daily[daily["Date"] > cutoff].copy()
    # Ensure we have train data
    if len(train) < 2:
        train = daily.iloc[:-1].copy()
        test = daily.iloc[-1:].copy()

    forecast_start = train["Date"].max() + pd.Timedelta(days=1)
    return daily, train, test, forecast_start


def train_prophet(
    train_df: pd.DataFrame,
    save_path: str = PROPHET_MODEL_PATH,
) -> "Prophet":
    """Train Prophet model on daily sales."""
    if not HAS_PROPHET:
        raise ImportError("Prophet is not installed. pip install prophet")
    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        interval_width=0.95,
    )
    model.fit(train_df[["ds", "y"]])
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w") as f:
        json.dump({"model": None}, f)
    # Prophet doesn't serialize easily to JSON; we save state via pickle for reload
    with open(save_path.replace(".json", ".pkl"), "wb") as f:
        pickle.dump(model, f)
    return model


def predict_prophet(
    model: "Prophet",
    periods: int,
    freq: str = "D",
) -> pd.DataFrame:
    """Generate forecast with Prophet."""
    future = model.make_future_dataframe(periods=periods, freq=freq)
    return model.predict(future)


def train_arima(
    train_series: pd.Series,
    order: tuple = (5, 1, 0),
    save_path: str = ARIMA_MODEL_PATH,
):
    """Train ARIMA on daily sales series."""
    if not HAS_ARIMA:
        raise ImportError("statsmodels is required for ARIMA. pip install statsmodels")
    model = ARIMA(train_series, order=order)
    fitted = model.fit()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "wb") as f:
        pickle.dump(fitted, f)
    return fitted


def predict_arima(fitted, steps: int) -> np.ndarray:
    """Forecast next steps with ARIMA."""
    return fitted.forecast(steps=steps)


def train_linear_regression(
    train_df: pd.DataFrame,
    save_path: str = LR_MODEL_PATH,
) -> LinearRegression:
    """
    Baseline: Linear Regression using month, year, weekday, quarter as features.
    """
    features = ["month", "year", "weekday", "quarter"]
    X = train_df[features].copy()
    y = train_df["y"]
    model = LinearRegression()
    model.fit(X, y)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "wb") as f:
        pickle.dump({"model": model, "features": features}, f)
    return model


def predict_linear_regression(
    model: LinearRegression,
    future_dates,
) -> np.ndarray:
    """Predict using LR with date-derived features. future_dates: DatetimeIndex or Series."""
    with open(LR_MODEL_PATH, "rb") as f:
        obj = pickle.load(f)
    model, features = obj["model"], obj["features"]
    dt = pd.to_datetime(future_dates)
    if isinstance(dt, pd.Series):
        month, year = dt.dt.month, dt.dt.year
        weekday, quarter = dt.dt.weekday, dt.dt.quarter
    else:
        month, year = dt.month, dt.year
        weekday, quarter = dt.weekday, dt.quarter
    X = pd.DataFrame({"month": month, "year": year, "weekday": weekday, "quarter": quarter})
    return model.predict(X[features])


# -----------------------------------------------------------------------------
# Full training and evaluation pipeline
# -----------------------------------------------------------------------------
def run_training_and_evaluation(
    forecast_months: int = FORECAST_HORIZON_MONTHS,
    test_months: int = TEST_SIZE_MONTHS,
) -> dict:
    """
    Train all models, evaluate on held-out test period, generate forecasts.
    Returns dict with metrics and forecast DataFrames.
    """
    daily, train, test, forecast_start = _get_series_and_cutoff(forecast_months)
    results = {"metrics": {}, "forecasts": {}}

    # Number of days to forecast (for test evaluation and future forecast)
    test_days = (test["Date"].max() - test["Date"].min()).days + 1
    forecast_days = min(forecast_months * 31, 365)

    # ----- Prophet -----
    if HAS_PROPHET:
        try:
            prophet_model = train_prophet(train, PROPHET_MODEL_PATH)
            prophet_forecast = predict_prophet(prophet_model, periods=forecast_days)
            # Align forecast to test period for evaluation (merge on ds)
            prophet_forecast["ds"] = pd.to_datetime(prophet_forecast["ds"]).dt.normalize()
            test_merge = test[["ds", "y"]].copy()
            test_merge["ds"] = pd.to_datetime(test_merge["ds"]).dt.normalize()
            pred_merged = test_merge.merge(
                prophet_forecast[["ds", "yhat"]],
                on="ds",
                how="left",
            ).dropna()
            if len(pred_merged) > 0:
                results["metrics"]["Prophet"] = compute_metrics(
                    pred_merged["y"].values, pred_merged["yhat"].values
                )
            else:
                results["metrics"]["Prophet"] = {"MAE": None, "RMSE": None, "error": "No overlapping dates"}
            results["forecasts"]["Prophet"] = prophet_forecast
        except Exception as e:
            results["metrics"]["Prophet"] = {"MAE": None, "RMSE": None, "error": str(e)}
            results["forecasts"]["Prophet"] = None
    else:
        results["metrics"]["Prophet"] = {"MAE": None, "RMSE": None, "error": "Prophet not installed"}
        results["forecasts"]["Prophet"] = None

    # ----- ARIMA -----
    if HAS_ARIMA:
        try:
            train_series = train.set_index("Date")["y"]
            order = (2, 1, 0) if len(train_series) < 60 else (5, 1, 0)
            if len(train_series) < 5:
                raise ValueError("Insufficient data for ARIMA")
            arima_fitted = train_arima(train_series, order=order, save_path=ARIMA_MODEL_PATH)
            arima_pred = predict_arima(arima_fitted, steps=min(len(test), len(train_series)))
            n_eval = min(len(test), len(arima_pred))
            results["metrics"]["ARIMA"] = compute_metrics(
                test["y"].values[:n_eval], arima_pred[:n_eval]
            )
            future_idx = pd.date_range(start=forecast_start, periods=forecast_days, freq="D")
            arima_future = predict_arima(arima_fitted, steps=forecast_days)
            results["forecasts"]["ARIMA"] = pd.DataFrame({"ds": future_idx, "yhat": arima_future})
        except Exception as e:
            results["metrics"]["ARIMA"] = {"MAE": None, "RMSE": None, "error": str(e)[:80]}
            results["forecasts"]["ARIMA"] = None
    else:
        results["metrics"]["ARIMA"] = {"MAE": None, "RMSE": None, "error": "statsmodels not installed"}
        results["forecasts"]["ARIMA"] = None

    # ----- Linear Regression -----
    try:
        lr_model = train_linear_regression(train, LR_MODEL_PATH)
        lr_pred = predict_linear_regression(lr_model, test["Date"])
        results["metrics"]["LinearRegression"] = compute_metrics(test["y"].values, lr_pred)
        future_dates = pd.date_range(start=forecast_start, periods=forecast_days, freq="D")
        lr_future = predict_linear_regression(lr_model, future_dates)
        results["forecasts"]["LinearRegression"] = pd.DataFrame({
            "ds": future_dates,
            "yhat": lr_future,
        })
    except Exception as e:
        results["metrics"]["LinearRegression"] = {"MAE": None, "RMSE": None, "error": str(e)}
        results["forecasts"]["LinearRegression"] = None

    # Save combined forecast (use best available model)
    _save_forecast_results(results, forecast_start, forecast_days)
    return results


def _save_forecast_results(results: dict, forecast_start: datetime, forecast_days: int):
    """Write forecast to CSV for dashboard."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    # Prefer Prophet if available, else ARIMA, else LR
    for name in ["Prophet", "ARIMA", "LinearRegression"]:
        fc = results["forecasts"].get(name)
        if fc is not None and "ds" in fc.columns and "yhat" in fc.columns:
            out = fc[["ds", "yhat"]].copy()
            out.columns = ["date", "forecast_sales"]
            out.to_csv(FORECAST_CSV_PATH, index=False)
            break


def load_forecast_csv() -> pd.DataFrame:
    """Load saved forecast for dashboard."""
    if os.path.exists(FORECAST_CSV_PATH):
        df = pd.read_csv(FORECAST_CSV_PATH, parse_dates=["date"])
        return df
    return pd.DataFrame()


if __name__ == "__main__":
    print("Running training and evaluation...")
    res = run_training_and_evaluation(forecast_months=6, test_months=3)
    for model, metrics in res["metrics"].items():
        print(f"{model}: {metrics}")
