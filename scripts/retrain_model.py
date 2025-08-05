# scripts/retrain_model.py

import pandas as pd
from prophet import Prophet
from datetime import datetime, timedelta
import os

DATA_PATH = "../data/tsa_daily_full.csv"
FORECAST_PATH = "../data/tsa_forecast.csv"
WEEKLY_HISTORY_PATH = "../data/weekly_forecast_history.csv"

# === LOAD DATA ===
df = pd.read_csv(DATA_PATH, parse_dates=['date'])
df = df[df['date'] >= '2023-01-01']  # Optional: exclude COVID-era noise

# === PREPARE FOR PROPHET ===
prophet_df = df.rename(columns={'date': 'ds', 'throughput': 'y'})

# === TRAIN MODEL ===
model = Prophet(daily_seasonality=True, yearly_seasonality=True)
model.add_country_holidays(country_name='US')
model.fit(prophet_df)

# === MAKE FUTURE FORECAST ===
future = model.make_future_dataframe(periods=30)  # Predict 30 days ahead
forecast = model.predict(future)

# === SAVE FULL FORECAST ===
forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].to_csv(FORECAST_PATH, index=False)
print(f"✅ Forecast saved to {FORECAST_PATH} ({len(forecast)} rows)")

# === NEW: LOG ONE WEEK OF FORECAST ===
# Only log this week's forecast if today is Monday
today = datetime.now().date()
if today.weekday() == 0:  # Monday = 0
    next_monday = today
    next_sunday = next_monday + timedelta(days=6)

    forecast_week = forecast[(forecast['ds'] >= pd.Timestamp(next_monday)) &
                             (forecast['ds'] <= pd.Timestamp(next_sunday))].copy()
    
    forecast_week['date_made'] = pd.Timestamp(today)

    if os.path.exists(WEEKLY_HISTORY_PATH):
        history_df = pd.read_csv(WEEKLY_HISTORY_PATH, parse_dates=['ds', 'date_made'])
        updated_df = pd.concat([history_df, forecast_week[['ds', 'yhat', 'date_made']]], ignore_index=True)
    else:
        updated_df = forecast_week[['ds', 'yhat', 'date_made']]

    updated_df.to_csv(WEEKLY_HISTORY_PATH, index=False)
    print(f"🗂️ Weekly forecast logged to {WEEKLY_HISTORY_PATH} ({len(forecast_week)} rows)")
else:
    print("ℹ️ Not Monday — skipping weekly forecast logging.")
