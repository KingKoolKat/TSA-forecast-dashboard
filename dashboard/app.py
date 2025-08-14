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

    
    st.plotly_chart(fig, use_container_width=True)

    # Show accuracy
    st.markdown(f"**✅ Model Accuracy on Completed Weeks:** {average_accuracy:.2f}%")


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

# === NEW: Kalshi Picks Section (robust + diagnostics) ===
import os, sys
import pandas as pd
import streamlit as st

# If kalshi_fetch.py lives next to app.py, this import will work:
from kalshi_fetch import (
    evaluate_above_lines,
    week_monday_from_now,
    fetch_tsa_contracts,           # new: so we can debug markets
)

st.markdown("---")
st.subheader("📈 Kalshi Picks — Expected Value (ABOVE X)")

# Controls
fee = st.number_input("Fee per share (USD)", value=0.00, step=0.01, key="kalshi_fee")
safety = st.slider("Uncertainty multiplier (conservative)", 1.0, 1.5, 1.10, 0.01, key="kalshi_safety")

# Pick a Monday snapshot that actually exists in history
# 1) load history once here for checking
hist = pd.read_csv("data/weekly_forecast_history.csv", parse_dates=["ds", "date_made"])
hist["ds"] = pd.to_datetime(hist["ds"]).dt.normalize()
hist["date_made"] = pd.to_datetime(hist["date_made"]).dt.normalize()

requested_mon = week_monday_from_now()
# All Mondays we have snapshots for (7 rows each)
mondays_available = sorted(hist["date_made"].dropna().unique())

# Choose the latest Monday <= requested
chosen_mon = None
for m in reversed(mondays_available):
    if pd.Timestamp(m) <= requested_mon:
        chosen_mon = pd.Timestamp(m)
        break

if chosen_mon is None and len(mondays_available) > 0:
    chosen_mon = pd.Timestamp(mondays_available[-1])  # fallback to latest overall

if chosen_mon is None:
    st.error("No weekly forecast snapshots found in weekly_forecast_history.csv. "
             "Run retrain_model.py on a Monday to create one.")
else:
    # Count rows for that chosen week
    wk_mask = (hist["date_made"] == chosen_mon) & \
              (hist["ds"] >= chosen_mon) & \
              (hist["ds"] <= chosen_mon + pd.Timedelta(days=6))
    rows_this_week = hist.loc[wk_mask].shape[0]

    if chosen_mon != requested_mon:
        st.info(f"Using committed forecast from **{chosen_mon.date()}** "
                f"(no snapshot found for {requested_mon.date()}).")
    else:
        st.caption(f"Using committed forecast made on Monday {chosen_mon.date()}.")

    # Quick diagnostic: show rows found
    with st.expander("Diagnostic: snapshot rows for chosen week"):
        st.write(f"Rows found for {chosen_mon.date()} week: {rows_this_week}")
        st.dataframe(hist.loc[wk_mask].sort_values("ds").head(10))

    try:
        # Fetch TSA markets first so we can show what we got even if EV fails
        markets_preview = fetch_tsa_contracts()
        with st.expander("Diagnostic: fetched TSA markets (top 10)"):
            st.write(len(markets_preview), "markets fetched")
            if markets_preview:
                st.write([{"ticker": m.get("ticker"), "title": m.get("title")} for m in markets_preview[:10]])

        # Compute EVs (this calls into the same fetch again internally)
        ev_table = evaluate_above_lines(
            history_csv="data/weekly_forecast_history.csv",
            week_monday=chosen_mon,
            fee_per_share=fee,
            safety_scale=safety,
        )

        if ev_table.empty:
            st.warning("No Kalshi TSA markets with usable prices found right now, or probability calc had no snapshot.")
        else:
            # === styling & filters (as before) ===
            col1, col2 = st.columns([1, 2])
            with col1:
                only_pos = st.checkbox("Show only positive EV", value=True, key="kalshi_only_pos")
            with col2:
                min_ev = st.number_input("Min best EV ($/share)", value=0.01 if only_pos else 0.00, step=0.01, key="kalshi_min_ev")

            df_view = ev_table.copy()
            cols = [
                "label", "ticker", "threshold",
                "yes_price", "no_price",
                "p_above_model", "EV_yes_$", "EV_no_$",
                "best_action", "best_EV_$"
            ]
            df_view = df_view[cols]

            if only_pos:
                df_view = df_view[df_view["best_EV_$"] > 0]
            if min_ev > 0:
                df_view = df_view[df_view["best_EV_$"] >= float(min_ev)]

            # number formatting
            if not df_view.empty:
                df_view["threshold"] = df_view["threshold"].round(0).astype("Int64")
                for c in ["yes_price", "no_price", "p_above_model", "EV_yes_$", "EV_no_$", "best_EV_$"]:
                    if c in df_view.columns:
                        df_view[c] = df_view[c].astype(float).round(4)

                # style: green/red on best action & EV
                import pandas as pd
                def _highlight_row(row: pd.Series):
                    styles = [''] * len(row)
                    ev = None
                    try:
                        ev = float(row.get("best_EV_$", None))
                    except Exception:
                        pass
                    color = "#16a34a" if (ev is not None and ev > 0) else "#ef4444"
                    for name in ["best_action", "best_EV_$"]:
                        if name in row.index:
                            idx = row.index.get_loc(name)
                            styles[idx] = f"font-weight:bold;color:white;background-color:{color}"
                    return styles

                styled = df_view.style.apply(_highlight_row, axis=1).format({
                    "yes_price": "{:.4f}",
                    "no_price": "{:.4f}",
                    "p_above_model": "{:.3%}",
                    "EV_yes_$": "${:.4f}",
                    "EV_no_$": "${:.4f}",
                    "best_EV_$": "${:.4f}",
                    "threshold": "{:,}",
                })
                st.dataframe(styled, use_container_width=True)
            else:
                st.info("No lines meet your EV filter.")
    except Exception as e:
        st.error(f"Could not compute EVs: {e}")
