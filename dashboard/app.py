# dashboard/app.py

import pandas as pd
import streamlit as st
import plotly.express as px
from datetime import datetime

# === CONFIG ===
st.set_page_config(page_title="TSA Forecast Dashboard", layout="wide")
st.title("✈️ TSA Passenger Forecast Dashboard")

# === LOAD DATA ===
df = pd.read_csv("data/tsa_daily_full.csv", parse_dates=['date'])
df = df[df['date'] >= '2023-01-01']
forecast = pd.read_csv("data/tsa_forecast.csv", parse_dates=['ds'])

# === MERGE ACTUALS & PREDICTIONS ===
pred_df = forecast.rename(columns={'ds': 'date'})
pred_df = pred_df.merge(df[['date', 'throughput']], on='date', how='left')

# === TOGGLE BETWEEN DAILY AND WEEKLY VIEW ===
view = st.radio("Select View Mode", ["Daily", "Weekly Averages"])

if view == "Daily":
    st.subheader("📈 Daily Throughput: Actual vs Predicted")
    fig = px.line(pred_df, x='date', y=['yhat', 'throughput'],
                  labels={'value': 'Passengers', 'variable': 'Legend'},
                  title="Daily TSA Passenger Count (Predicted vs Actual)")
    st.plotly_chart(fig, use_container_width=True)

elif view == "Weekly Averages":
    st.subheader("📊 Weekly Averages: Actual vs Predicted")
    pred_df['week'] = pred_df['date'].dt.to_period('W').apply(lambda r: r.start_time)
    weekly = pred_df.groupby('week').agg(
        predicted_avg=('yhat', 'mean'),
        actual_avg=('throughput', 'mean')
    ).reset_index()

    fig = px.bar(weekly, x='week', y=['predicted_avg', 'actual_avg'],
                 barmode='group',
                 labels={'value': 'Passengers', 'variable': 'Legend'},
                 title="Weekly Average TSA Throughput")
    st.plotly_chart(fig, use_container_width=True)

# === CURRENT WEEK FORECAST ===
today = datetime.now()
start_of_week = today - pd.Timedelta(days=today.weekday())
end_of_week = start_of_week + pd.Timedelta(days=6)

this_week = pred_df[(pred_df['date'] >= start_of_week) & (pred_df['date'] <= end_of_week)]
known_days = this_week[this_week['throughput'].notna()]
predicted_days = this_week[this_week['throughput'].isna()]

blended_total = known_days['throughput'].sum() + predicted_days['yhat'].sum()
blended_avg = blended_total / 7

week_range_label = f"{start_of_week.strftime('%b %d')} – {end_of_week.strftime('%b %d')}"

st.markdown("---")
st.subheader("📅 Current Week Avg (Blended)")
st.metric(f"{week_range_label} Avg (Live)", f"{blended_avg:,.0f} passengers")
