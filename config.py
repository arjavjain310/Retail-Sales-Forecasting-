"""
Configuration and constants for Retail Sales Forecasting project.
Centralizes paths, URLs, and default parameters for easy maintenance.
"""

import os

# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RAW_DATA_PATH = os.path.join(DATA_DIR, "raw")
PROCESSED_DATA_PATH = os.path.join(DATA_DIR, "processed")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")

# Create directories if they don't exist
for path in [DATA_DIR, RAW_DATA_PATH, PROCESSED_DATA_PATH, MODELS_DIR, RESULTS_DIR]:
    os.makedirs(path, exist_ok=True)

# -----------------------------------------------------------------------------
# Data source (Plotly Supermarket Sales - real-world retail dataset)
# -----------------------------------------------------------------------------
DATA_URL = "https://raw.githubusercontent.com/plotly/datasets/master/supermarket_Sales.csv"
RAW_CSV_FILENAME = "supermarket_sales.csv"
RAW_CSV_PATH = os.path.join(RAW_DATA_PATH, RAW_CSV_FILENAME)
PROCESSED_CSV_PATH = os.path.join(PROCESSED_DATA_PATH, "sales_processed.csv")
DAILY_AGGREGATE_PATH = os.path.join(PROCESSED_DATA_PATH, "sales_daily.csv")

# -----------------------------------------------------------------------------
# Currency (Indian Rupee)
# -----------------------------------------------------------------------------
CURRENCY_SYMBOL = "₹"
# Convert dataset values to INR if original is in USD (e.g. 1 USD ≈ 83 INR)
USD_TO_INR = 83.0

# -----------------------------------------------------------------------------
# Column mappings (align with dataset)
# -----------------------------------------------------------------------------
DATE_COL = "Date"
SALES_COL = "Total"          # Revenue per transaction
CATEGORY_COL = "Product line"
QUANTITY_COL = "Quantity"


def format_inr(value: float) -> str:
    """Format number as Indian Rupee (e.g. ₹1,25,000)."""
    s = str(int(round(value)))
    if len(s) <= 3:
        return f"{CURRENCY_SYMBOL}{s}"
    r = s[-3:]
    s = s[:-3]
    while s:
        r = s[-2:] + "," + r
        s = s[:-2]
    return f"{CURRENCY_SYMBOL}{r}"

# -----------------------------------------------------------------------------
# Forecasting defaults
# -----------------------------------------------------------------------------
FORECAST_HORIZON_MONTHS = 6
TEST_SIZE_MONTHS = 3        # Last N months for validation
EVAL_METRICS = ["MAE", "RMSE"]

# -----------------------------------------------------------------------------
# Model artifact names
# -----------------------------------------------------------------------------
PROPHET_MODEL_PATH = os.path.join(MODELS_DIR, "prophet_model.json")
ARIMA_MODEL_PATH = os.path.join(MODELS_DIR, "arima_model.pkl")
LR_MODEL_PATH = os.path.join(MODELS_DIR, "lr_model.pkl")
FORECAST_CSV_PATH = os.path.join(RESULTS_DIR, "forecast_results.csv")
