"""
One-shot pipeline: download data, preprocess, train models, and save forecasts.
Run this before opening the dashboard to ensure forecasts are available.
Usage: python run_pipeline.py
"""

import sys
import os

# Ensure project root is on path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from config import RAW_CSV_PATH, PROCESSED_CSV_PATH, RESULTS_DIR
from src.data_preprocessing import download_data, preprocess_pipeline
from src.model_training import run_training_and_evaluation
from src.visualization import create_eda_figures


def main():
    print("1. Downloading data...")
    download_data(save_path=RAW_CSV_PATH)

    print("2. Preprocessing and feature engineering...")
    df = preprocess_pipeline(save_processed=True, save_daily=True)
    print(f"   Processed {len(df)} rows. Date range: {df['Date'].min()} to {df['Date'].max()}")

    print("3. Generating EDA figures...")
    create_eda_figures(df, output_dir=RESULTS_DIR)
    print(f"   Figures saved to {RESULTS_DIR}")

    print("4. Training models and evaluating (MAE / RMSE)...")
    results = run_training_and_evaluation(forecast_months=6, test_months=3)
    for model_name, metrics in results["metrics"].items():
        if isinstance(metrics, dict) and metrics.get("MAE") is not None:
            print(f"   {model_name}: MAE = {metrics['MAE']:.2f}, RMSE = {metrics['RMSE']:.2f}")
        else:
            print(f"   {model_name}: {metrics.get('error', metrics)}")

    print("5. Done. Start the dashboard with: streamlit run app.py")


if __name__ == "__main__":
    main()
