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

import math
import datetime as dt
import pandas as pd
import requests
import streamlit as st

# --- Helper: from Prophet's 80% band to daily sigma, then weekly avg prob ---
Z80 = 1.2815515655446004  # central 80% interval z-score (10th..90th)

def prob_weekly_avg_above_threshold(yhat, yhat_lower, yhat_upper, strike_T):
    """
    yhat/yhat_lower/yhat_upper: lists of 7 daily values (aligned to the Kalshi week)
    strike_T: numeric threshold (e.g., 2_800_000)
    returns: P(weekly average > strike_T) under a normal approx
    """
    assert len(yhat) == len(yhat_lower) == len(yhat_upper) == 7, "Need exactly 7 aligned days"
    mus = [float(m) for m in yhat]
    sigmas = [ (u - l) / (2.0 * Z80) for l, u in zip(yhat_lower, yhat_upper) ]
    mu_avg = sum(mus) / 7.0
    var_avg = sum(s**2 for s in sigmas) / 49.0  # Var(mean) = (1/49)*sum(sigma_i^2)
    sigma_avg = max(var_avg, 0.0) ** 0.5 or 1e-9  # guard tiny variance
    z = (strike_T - mu_avg) / sigma_avg
    cdf = 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))  # Φ(z)
    p = 1.0 - cdf  # P(avg > T)
    return max(0.0, min(1.0, p))

# --- Helper: parse Kalshi event_ticker to get the "week ending" date ---
# Examples: "KXTSAW-25AUG24" -> 2025-08-24
_MONTHS = {"JAN":1,"FEB":2,"MAR":3,"APR":4,"MAY":5,"JUN":6,"JUL":7,"AUG":8,"SEP":9,"OCT":10,"NOV":11,"DEC":12}
def week_end_date_from_event_ticker(event_ticker: str) -> dt.date:
    # Expect suffix like YYMONDD (e.g., 25AUG24)
    suffix = event_ticker.rsplit("-", 1)[-1]
    yy = int(suffix[0:2])
    mon_txt = suffix[2:5].upper()
    dd = int(suffix[5:7])
    year = 2000 + yy  # Kalshi uses 20YY
    month = _MONTHS[mon_txt]
    return dt.date(year, month, dd)

def current_week_end_central() -> dt.date:
    # Use America/Chicago “today” to compute the upcoming Sunday
    today_cdt = pd.Timestamp("now", tz="America/Chicago").date()
    days_until_sun = (6 - today_cdt.weekday()) % 7
    return today_cdt + dt.timedelta(days=days_until_sun)


# --- Helper: get the 7 days ending on week_end (inclusive) from your forecast DF ---
def get_week_slice(df_daily: pd.DataFrame, week_end: dt.date):
    """
    df_daily: DataFrame with columns ['ds','yhat','yhat_lower','yhat_upper']
              'ds' should be datetime-like (date or timestamp)
    week_end: datetime.date representing the market's "week ending" date
    returns: (yhat_list, lower_list, upper_list) for the 7-day window
    """
    start = pd.to_datetime(week_end) - pd.Timedelta(days=6)
    end = pd.to_datetime(week_end)
    mask = (pd.to_datetime(df_daily["ds"]).dt.normalize() >= start.normalize()) & \
           (pd.to_datetime(df_daily["ds"]).dt.normalize() <= end.normalize())
    wk = df_daily.loc[mask].sort_values("ds")
    if len(wk) != 7:
        raise ValueError(f"Need 7 forecast days for {start.date()}..{end.date()}, found {len(wk)}")
    return wk["yhat"].tolist(), wk["yhat_lower"].tolist(), wk["yhat_upper"].tolist()

# ===============================
# 📊 TSA Forecast Probabilities (Prophet-based)
# ===============================

st.subheader("📊 TSA Forecast Probabilities (Prophet → weekly average)")

# You likely already have this in memory; adjust the variable name if needed.
# Expected columns: ['ds', 'yhat', 'yhat_lower', 'yhat_upper'] with daily rows.
df_daily = history.copy()
# Ensure correct dtypes
df_daily['ds'] = pd.to_datetime(df_daily['ds'], errors='coerce')
for col in ['yhat','yhat_lower','yhat_upper']:
    df_daily[col] = pd.to_numeric(df_daily[col], errors='coerce')

# Fetch active TSA markets for the series (public; no auth)
try:
    series_ticker = "KXTSAW"  # TSA weekly average series
    url = f"https://api.elections.kalshi.com/trade-api/v2/markets?series_ticker={series_ticker}"
    r = requests.get(url, timeout=10)
    r.raise_for_status()
    markets = r.json().get("markets", [])
    if not markets:
        st.info("No active TSA markets returned.")
    else:
        # Show a compact table of strike + Prophet probability
        rows = []
        for m in markets:
            try:
                week_end = week_end_date_from_event_ticker(m["event_ticker"])
                if week_end != cur_week_end:
                    continue  # only show current week markets
        
                yhat, yl, yu = get_week_slice(df_daily, week_end)
                p = prob_weekly_avg_above_threshold(yhat, yl, yu, m["floor_strike"])

                yes_ask = m.get("yes_ask")
                no_ask  = m.get("no_ask")

                ev_yes_cents = None if yes_ask is None else (100.0 * p - float(yes_ask))
                ev_no_cents  = None if no_ask  is None else (100.0 * (1.0 - p) - float(no_ask))


                rows.append({
                    "Market": m.get("ticker", ""),
                    "Week Ending": week_end.isoformat(),
                    "Strike": int(m["floor_strike"]),
                    "Prophet P(avg>strike)": round(p, 4),
                    # Optional: include quotes for context (in cents)
                    "Yes Bid": m.get("yes_bid"),
                    "Yes Ask": m.get("yes_ask"),
                    "No Bid": m.get("no_bid"),
                    "No Ask": m.get("no_ask"),
                    "EV Yes @ Ask (¢)": round(ev_yes_cents, 2),
                    "EV No @ Ask (¢)": round(ev_no_cents, 2),
                    "EV Yes @ Ask ($)": round(ev_yes_cents / 100.0, 4),
                    "EV No @ Ask ($)": round(ev_no_cents  / 100.0, 4),

                })
            except Exception as e:
                # If any single market can't be aligned, skip and continue
                rows.append({
                    "Market": m.get("ticker", ""),
                    "Week Ending": "(unmatched)",
                    "Strike": m.get("floor_strike"),
                    "Prophet P(avg>strike)": None,
                    "Error": str(e)[:120]
                })

        st.dataframe(pd.DataFrame(rows).sort_values(["Week Ending","Strike"]))
except Exception as e:
    st.error(f"Failed to fetch TSA markets or compute probabilities: {e}")

