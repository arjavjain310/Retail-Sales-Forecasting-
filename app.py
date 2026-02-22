"""
Retail Mart Sales Dashboard - Interactive Streamlit app.
Currency: Indian Rupee (₹). Shows monthly & daily sales, category dropdown,
reports, and rich sidebar. Dataset: retail store / supermarket sales.
"""

import os
import sys
from datetime import datetime, timedelta

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config import (
    DATE_COL,
    SALES_COL,
    CATEGORY_COL,
    FORECAST_CSV_PATH,
    CURRENCY_SYMBOL,
    USD_TO_INR,
    format_inr,
)
from src.data_preprocessing import get_processed_data, get_daily_data, download_data
from src.model_training import run_training_and_evaluation

st.set_page_config(
    page_title="Retail Mart Sales Dashboard",
    page_icon="🏪",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_data(ttl=3600)
def load_data():
    try:
        download_data()
    except Exception:
        pass
    df = get_processed_data()
    # Convert to INR for display (dataset values treated as USD)
    df = df.copy()
    df["sales_inr"] = df[SALES_COL] * USD_TO_INR
    return df


@st.cache_data(ttl=3600)
def load_daily():
    d = get_daily_data()
    d = d.copy()
    d["sales_inr"] = d["sales"] * USD_TO_INR
    return d


def load_forecast():
    if os.path.exists(FORECAST_CSV_PATH):
        df = pd.read_csv(FORECAST_CSV_PATH, parse_dates=["date"])
        df["forecast_sales_inr"] = df["forecast_sales"] * USD_TO_INR
        return df
    return None


def main():
    st.title("🏪 Retail Mart Sales Dashboard")
    st.markdown("**Currency: Indian Rupee (₹)** · Explore sales, categories, and forecasts.")

    try:
        df = load_data()
    except Exception as e:
        st.error(f"Could not load data. Error: {e}")
        return

    df[DATE_COL] = pd.to_datetime(df[DATE_COL])
    min_date = df[DATE_COL].min().date()
    max_date = df[DATE_COL].max().date()

    # ----- SIDEBAR -----
    st.sidebar.header("🎛️ Filters & Options")

    # Quick date presets
    st.sidebar.subheader("Quick date range")
    preset = st.sidebar.selectbox(
        "Preset",
        ["Custom", "Last 7 days", "Last 30 days", "This month", "Last 3 months", "Last 6 months", "All time"],
        key="preset",
    )
    if preset == "Last 7 days":
        start_date = max(min_date, max_date - timedelta(days=7))
        end_date = max_date
    elif preset == "Last 30 days":
        start_date = max(min_date, max_date - timedelta(days=30))
        end_date = max_date
    elif preset == "This month":
        start_date = max_date.replace(day=1)
        end_date = max_date
    elif preset == "Last 3 months":
        start_date = max(min_date, max_date - timedelta(days=90))
        end_date = max_date
    elif preset == "Last 6 months":
        start_date = max(min_date, max_date - timedelta(days=180))
        end_date = max_date
    elif preset == "All time":
        start_date, end_date = min_date, max_date
    else:
        date_range = st.sidebar.date_input(
            "Date range",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date,
            key="date_range",
        )
        if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
            start_date, end_date = date_range[0], date_range[1]
        else:
            start_date, end_date = min_date, max_date

    # Category dropdown
    st.sidebar.subheader("Category")
    categories = ["All Categories"] + sorted(df[CATEGORY_COL].dropna().unique().tolist())
    selected_category = st.sidebar.selectbox(
        "Select category",
        options=categories,
        key="category_dropdown",
    )
    if selected_category == "All Categories":
        selected_categories = categories[1:]  # all
    else:
        selected_categories = [selected_category]

    # Apply filters
    mask_date = (df[DATE_COL].dt.date >= start_date) & (df[DATE_COL].dt.date <= end_date)
    mask_cat = df[CATEGORY_COL].isin(selected_categories)
    filtered = df.loc[mask_date & mask_cat].copy()
    sales_col = "sales_inr"

    if filtered.empty:
        st.warning("No data for the selected filters. Change date range or category.")
        return

    # Sidebar: Summary for current filter
    st.sidebar.markdown("---")
    st.sidebar.subheader("📊 Current filter summary")
    st.sidebar.metric("Total sales (INR)", format_inr(filtered[sales_col].sum()))
    st.sidebar.metric("Transactions", f"{len(filtered):,}")
    st.sidebar.metric("Avg order (INR)", format_inr(filtered[sales_col].mean()))

    # Reports section in sidebar
    st.sidebar.markdown("---")
    st.sidebar.subheader("📋 Reports")
    if "saved_reports" not in st.session_state:
        st.session_state.saved_reports = []

    if st.sidebar.button("Save current view as report"):
        report = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "date_range": f"{start_date} to {end_date}",
            "category": selected_category,
            "total_sales_inr": filtered[sales_col].sum(),
            "transactions": len(filtered),
        }
        st.session_state.saved_reports.insert(0, report)
        st.session_state.saved_reports = st.session_state.saved_reports[:20]
        st.sidebar.success("Report saved.")

    if st.session_state.saved_reports:
        with st.sidebar.expander("View previous reports", expanded=False):
            for i, r in enumerate(st.session_state.saved_reports[:10]):
                st.markdown(f"**Report {i+1}** · {r['timestamp']}")
                st.caption(f"Period: {r['date_range']} · Category: {r['category']}")
                st.caption(f"Sales: {format_inr(r['total_sales_inr'])} · Txns: {r['transactions']:,}")
                st.markdown("---")
    else:
        st.sidebar.caption("No reports yet. Use 'Save current view as report' to add one.")

    # Export
    st.sidebar.markdown("---")
    st.sidebar.subheader("Export")
    csv = filtered[[DATE_COL, CATEGORY_COL, sales_col, "Quantity"]].to_csv(index=False)
    st.sidebar.download_button(
        "Download filtered data (CSV)",
        csv,
        file_name=f"retail_sales_{start_date}_{end_date}.csv",
        mime="text/csv",
        key="export_csv",
    )

    st.sidebar.markdown("---")
    st.sidebar.caption("Retail Mart · Data: store/supermarket sales. All amounts in ₹ (INR).")

    # ----- MAIN: KPIs -----
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Total revenue (₹)", format_inr(filtered[sales_col].sum()))
    with col2:
        st.metric("Transactions", f"{len(filtered):,}")
    with col3:
        st.metric("Avg order (₹)", format_inr(filtered[sales_col].mean()))
    with col4:
        monthly_total = filtered.set_index(DATE_COL)[sales_col].resample("ME").sum()
        st.metric("Total month sales (₹)", format_inr(monthly_total.sum()) if len(monthly_total) else "—")
    with col5:
        daily_avg = filtered.set_index(DATE_COL)[sales_col].resample("D").sum().mean()
        st.metric("Avg daily sales (₹)", format_inr(daily_avg) if not pd.isna(daily_avg) else "—")

    # ----- Total month sales & Daily sales -----
    st.subheader("📅 Total month sales & Daily sales")
    col_month, col_daily = st.columns(2)

    with col_month:
        monthly = filtered.set_index(DATE_COL)[sales_col].resample("ME").sum().reset_index()
        monthly.columns = [DATE_COL, "sales_inr"]
        fig_m = px.bar(monthly, x=DATE_COL, y="sales_inr", title="Total monthly sales (₹)")
        fig_m.update_layout(yaxis_title="Sales (₹)", xaxis_title="Month")
        st.plotly_chart(fig_m, use_container_width=True)

    with col_daily:
        daily_series = filtered.set_index(DATE_COL)[sales_col].resample("D").sum().reset_index()
        daily_series.columns = [DATE_COL, "sales_inr"]
        fig_d = px.line(daily_series, x=DATE_COL, y="sales_inr", title="Daily sales (₹)")
        fig_d.update_layout(yaxis_title="Sales (₹)", xaxis_title="Date")
        st.plotly_chart(fig_d, use_container_width=True)

    # ----- Tabs -----
    tab1, tab2, tab3, tab4 = st.tabs([
        "Sales trends",
        "Category revenue",
        "Monthly growth",
        "Forecast",
    ])

    with tab1:
        st.subheader("Sales over time")
        agg_freq = st.radio("View", ["Daily", "Monthly"], horizontal=True, key="agg")
        freq = "D" if agg_freq == "Daily" else "ME"
        trend = filtered.set_index(DATE_COL)[sales_col].resample(freq).sum().reset_index()
        trend.columns = [DATE_COL, "sales_inr"]
        fig = px.line(trend, x=DATE_COL, y="sales_inr", title="Sales trend (₹)")
        fig.update_layout(yaxis_title="Sales (₹)", xaxis_title="Date")
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.subheader("Revenue by category")
        cat_rev = filtered.groupby(CATEGORY_COL)[sales_col].sum().sort_values(ascending=True)
        fig = px.bar(
            x=cat_rev.values,
            y=cat_rev.index,
            orientation="h",
            labels={"x": "Revenue (₹)", "y": "Category"},
            title="Category-wise revenue (₹)",
        )
        st.plotly_chart(fig, use_container_width=True)
        fig2 = px.pie(values=cat_rev.values, names=cat_rev.index, title="Revenue share by category")
        st.plotly_chart(fig2, use_container_width=True)

    with tab3:
        st.subheader("Monthly sales growth")
        monthly = filtered.set_index(DATE_COL)[sales_col].resample("ME").sum().reset_index()
        monthly["growth_pct"] = monthly[sales_col].pct_change() * 100
        fig = go.Figure()
        fig.add_trace(
            go.Bar(
                x=monthly[DATE_COL],
                y=monthly[sales_col],
                name="Sales",
                marker_color=["green" if g >= 0 else "red" for g in monthly["growth_pct"].fillna(0)],
            )
        )
        fig.update_layout(
            title="Monthly sales (₹) — Green: growth, Red: decline",
            xaxis_title="Month",
            yaxis_title="Sales (₹)",
        )
        st.plotly_chart(fig, use_container_width=True)
        monthly["Month"] = monthly[DATE_COL].dt.to_period("M").astype(str)
        disp = monthly[["Month", sales_col, "growth_pct"]].copy()
        disp.columns = ["Month", "Sales (₹)", "Growth %"]
        disp["Sales (₹)"] = disp["Sales (₹)"].apply(lambda x: format_inr(x))
        disp["Growth %"] = disp["Growth %"].apply(lambda x: f"{x:.1f}%" if pd.notna(x) else "—")
        st.dataframe(disp, use_container_width=True, hide_index=True)

    with tab4:
        st.subheader("Sales forecast (₹)")
        forecast_df = load_forecast()
        if forecast_df is not None and not forecast_df.empty:
            fig = px.line(
                forecast_df,
                x="date",
                y="forecast_sales_inr",
                title="Forecasted sales (₹)",
            )
            fig.update_layout(xaxis_title="Date", yaxis_title="Forecast (₹)")
            st.plotly_chart(fig, use_container_width=True)
            daily = load_daily()
            if not daily.empty:
                daily = daily.rename(columns={"Date": "date"})
                fig2 = go.Figure()
                fig2.add_trace(
                    go.Scatter(x=daily["date"], y=daily["sales_inr"], name="Actual (historical)", mode="lines")
                )
                fig2.add_trace(
                    go.Scatter(
                        x=forecast_df["date"],
                        y=forecast_df["forecast_sales_inr"],
                        name="Forecast",
                        mode="lines",
                        line=dict(dash="dash"),
                    )
                )
                fig2.update_layout(title="Actual vs forecast (₹)", xaxis_title="Date", yaxis_title="Sales (₹)")
                st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("No forecast yet. Run training to generate forecasts.")
            if st.button("Run model training and generate forecast"):
                with st.spinner("Training models..."):
                    try:
                        res = run_training_and_evaluation(forecast_months=6, test_months=3)
                        st.success("Training complete.")
                        for model, m in res["metrics"].items():
                            if isinstance(m, dict) and m.get("MAE") is not None:
                                st.write(f"**{model}**: MAE = {m['MAE']:.2f}, RMSE = {m['RMSE']:.2f}")
                            else:
                                st.write(f"**{model}**: {m.get('error', m)}")
                        st.rerun()
                    except Exception as e:
                        st.error(str(e))


if __name__ == "__main__":
    main()
