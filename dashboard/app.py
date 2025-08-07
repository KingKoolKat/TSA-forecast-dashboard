import pandas as pd
import streamlit as st
import plotly.express as px
from datetime import datetime

# === CONFIG ===
st.set_page_config(page_title="TSA Forecast Dashboard", layout="wide")
st.title("✈️ TSA Passenger Forecast Dashboard")

# === LOAD DATA ===
df = pd.read_csv("data/tsa_daily_full.csv", parse_dates=["date"])
df = df[df["date"] >= "2023-01-01"]

history = pd.read_csv("data/weekly_forecast_history.csv", parse_dates=["date_made", "ds"])
history = history.sort_values("date_made")

# Merge actuals to historical forecasts
merged = history.merge(df.rename(columns={"date": "ds"}), on="ds", how="left")
merged["absolute_error"] = (merged["yhat"] - merged["throughput"]).abs()
merged["percent_error"] = (merged["absolute_error"] / merged["throughput"]) * 100

# === TOGGLE VIEW ===
view = st.radio("Select View Mode", ["Daily", "Weekly Averages"])

if view == "Daily":
    st.subheader("📈 Daily Forecasts vs Actuals (From Historical Model Runs)")
    fig = px.line(merged, x="ds", y=["yhat", "throughput"],
                  labels={"value": "Passengers", "variable": "Legend"},
                  title="Daily Forecast (Historical) vs Actuals")
    st.plotly_chart(fig, use_container_width=True)

    # Optional: accuracy metrics
    st.markdown("### 📏 Accuracy (All Time)")
    avg_mae = merged["absolute_error"].mean()
    avg_mape = merged["percent_error"].mean()
    st.write(f"**MAE:** {avg_mae:,.0f} passengers")
    st.write(f"**MAPE:** {avg_mape:.2f}%")

elif view == "Weekly Averages":
    st.subheader("📊 Weekly Averages: Forecast vs Actual")
    # Add 'week' column to merged data
    merged['week'] = merged['ds'].dt.to_period("W").apply(lambda r: r.start_time)

    # Identify current week
    today = datetime.now().date()
    current_week = today - pd.Timedelta(days=today.weekday())

    # Aggregate weekly averages
    weekly = merged.groupby('week').agg(
        predicted_avg=('yhat', 'mean'),
        actual_avg=('throughput', 'mean'),
        percent_error=('percent_error', 'mean')
    ).reset_index()

    # Mark current week
    weekly['is_current_week'] = weekly['week'] == pd.Timestamp(current_week)

    # Add display label
    weekly['label'] = weekly.apply(
        lambda row: f"{row['week'].strftime('%b %d')} (current)" if row['is_current_week'] 
                    else row['week'].strftime('%b %d'),
        axis=1
    )

    # Filter out current week for accuracy stats
    completed_weeks = weekly[~weekly['is_current_week']]
    average_accuracy = 100 - completed_weeks['percent_error'].mean()

    # Plot weekly bars
    fig = px.bar(weekly, x='label', y=['predicted_avg', 'actual_avg'],
                barmode='group',
                labels={'value': 'Passengers', 'variable': 'Legend'},
                title="Weekly Average TSA Throughput")

    st.subheader("📊 Weekly Averages: Actual vs Predicted")
    st.plotly_chart(fig, use_container_width=True)

    # Show accuracy
    st.markdown(f"**✅ Model Accuracy on Completed Weeks:** {average_accuracy:.2f}%")


# === CURRENT WEEK FORECAST (from committed forecast) ===
today = datetime.now()
start_of_week = today - pd.Timedelta(days=today.weekday())
end_of_week = start_of_week + pd.Timedelta(days=6)

current_week_prediction = history[
    (history["ds"] >= start_of_week) &
    (history["ds"] <= end_of_week) &
    (history["date_made"] == start_of_week)
]

if not current_week_prediction.empty:
    current_week_avg = current_week_prediction["yhat"].mean()
    label = f"{start_of_week.strftime('%b %d')} – {end_of_week.strftime('%b %d')}"
    st.markdown("---")
    st.subheader("📅 This Week's Forecast (As Predicted Monday)")
    st.metric(f"{label} Avg Forecast", f"{current_week_avg:,.0f} passengers")
