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

# === CURRENT WEEK FORECAST (from committed forecast) ===
now = pd.Timestamp.now()
monday = (now.normalize() - pd.to_timedelta(now.weekday(), unit="D"))  # start of this week at 00:00
sunday = monday + pd.Timedelta(days=6)

# Make sure the history columns are timestamps at midnight
history['ds'] = pd.to_datetime(history['ds']).dt.normalize()
history['date_made'] = pd.to_datetime(history['date_made']).dt.normalize()

mask = (
    (history['ds'] >= monday) &
    (history['ds'] <= sunday) &
    (history['date_made'] == monday)
)

current_week_prediction = history.loc[mask]

st.markdown("---")
st.subheader("📅 This Week's Forecast (As Predicted Monday)")

if not current_week_prediction.empty:
    current_week_avg = current_week_prediction['yhat'].mean()
    label = f"{monday.strftime('%b %d')} – {sunday.strftime('%b %d')}"
    st.metric(f"{label} Avg Forecast", f"{current_week_avg:,.0f} passengers")
else:
    # Debug helpers so you can see what the app thinks is available
    st.warning(
        "No forecast found for this week in the historical record. "
        "Check that `weekly_forecast_history.csv` has rows with "
        f"`date_made == {monday.date()}` and 7 entries where `ds` is between "
        f"{monday.date()} and {sunday.date()}."
    )
    with st.expander("Debug info"):
        st.write("Unique `date_made` values in history:", history['date_made'].dropna().unique())
        st.write("Most recent rows:", history.sort_values(['date_made','ds']).tail(10))


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

    
    st.plotly_chart(fig, use_container_width=True)

    # Show accuracy
    st.markdown(f"**✅ Model Accuracy on Completed Weeks:** {average_accuracy:.2f}%")



