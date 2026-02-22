# Retail Sales Forecasting & Trend Analysis

[![GitHub](https://img.shields.io/badge/GitHub-arjavjain310%2FRetail--Sales--Forecasting--blue)](https://github.com/arjavjain310/Retail-Sales-Forecasting-)

End-to-end project for retail sales forecasting and trend analysis using a real-world supermarket sales dataset. **Currency: Indian Rupee (₹).** Deploy on [Streamlit Community Cloud](https://share.streamlit.io). Includes data cleaning, feature engineering, exploratory data analysis, time series forecasting (ARIMA, Prophet, Linear Regression), and an interactive Streamlit dashboard.

## Project Structure

```
Retail Sales Forecasting/
├── app.py                 # Streamlit dashboard (run with: streamlit run app.py)
├── config.py              # Paths, URLs, and constants
├── run_pipeline.py        # Full pipeline: download → preprocess → train → EDA
├── requirements.txt       # Python dependencies
├── README.md
├── data/
│   ├── raw/               # Raw CSV (downloaded)
│   └── processed/         # Cleaned data and daily aggregates
├── models/                # Saved models (Prophet, ARIMA, LR)
├── results/               # Forecast CSV and EDA figures
└── src/
    ├── __init__.py
    ├── data_preprocessing.py   # Load, clean, feature engineering
    ├── model_training.py       # ARIMA, Prophet, LR; MAE/RMSE evaluation
    └── visualization.py       # EDA and trend plots
```

## Dataset

The project uses the **Supermarket Sales** dataset (Plotly). It is downloaded automatically from:

- `https://raw.githubusercontent.com/plotly/datasets/master/supermarket_Sales.csv`

Columns include: Invoice ID, Branch, City, Customer type, Gender, **Product line**, Unit price, **Quantity**, **Total** (revenue), **Date**, Time, Payment, etc. Product line is used as category; Total as sales.

## Setup

1. **Create a virtual environment (recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate   # Windows: venv\Scripts\activate
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the full pipeline (download data, preprocess, train models, generate EDA):**
   ```bash
   python run_pipeline.py
   ```

4. **Launch the dashboard:**
   ```bash
   streamlit run app.py
   ```

## Features

- **Data preprocessing:** Parsing dates, handling missing values, feature engineering (month, year, weekday, quarter, season).
- **EDA:** Sales trends, monthly growth, category-wise revenue, revenue distribution, seasonality and weekday patterns.
- **Forecasting:** Prophet, ARIMA, and Linear Regression; evaluation with **MAE** and **RMSE**; forecast for the next **3–6 months**.
- **Dashboard:** Filter by **date range** and **product category**; view sales trends, category revenue, monthly growth charts, and forecasted sales. Option to run training from the app if no forecast exists.

## Metrics

Models are evaluated on a held-out period (last 3 months by default) using:

- **MAE** (Mean Absolute Error)
- **RMSE** (Root Mean Squared Error)

## Requirements

- Python 3.8+
- See `requirements.txt` for package versions (pandas, numpy, matplotlib, seaborn, plotly, statsmodels, prophet, scikit-learn, streamlit, requests).
- **Prophet** is optional; if not installed, ARIMA and Linear Regression are still used. Install with `pip install prophet` for Prophet support.

## GitHub & Deploy

- **Repository:** [https://github.com/arjavjain310/Retail-Sales-Forecasting-](https://github.com/arjavjain310/Retail-Sales-Forecasting-)
- **Deploy on Streamlit Cloud:** Go to [share.streamlit.io](https://share.streamlit.io) → New app → select this repo → Main file: `app.py` → Deploy. See [DEPLOY.md](DEPLOY.md) for steps.

## Resume-Ready Notes

- Modular layout: preprocessing, modeling, and visualization are in separate modules.
- Configuration centralized in `config.py`.
- Comments and docstrings throughout.
- Single-command pipeline and clear README for reproducibility.
