# scripts/update_data.py

import requests
import pandas as pd
from bs4 import BeautifulSoup
from datetime import datetime

# === CONFIG ===
TSA_URL = "https://www.tsa.gov/travel/passenger-volumes"
DATA_PATH = "data/tsa_daily_full.csv"

# === FETCH TODAY'S TSA DATA ===
def fetch_latest_tsa_row():
    resp = requests.get(TSA_URL)
    soup = BeautifulSoup(resp.text, "lxml")
    table = soup.find("table")
    df = pd.read_html(str(table))[0]
    df = df.rename(columns={
        'Date': 'date',
        'Numbers': 'throughput'
    })
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')
    return df.tail(1)  # Get most recent row

# === LOAD EXISTING DATA ===
def load_existing_data():
    return pd.read_csv(DATA_PATH, parse_dates=['date'])

# === APPEND IF NEW ===
def update_data():
    new_row = fetch_latest_tsa_row()
    existing = load_existing_data()

    if new_row['date'].iloc[0] > existing['date'].max():
        updated = pd.concat([existing, new_row], ignore_index=True)
        updated.to_csv(DATA_PATH, index=False)
        print(f"✅ New row added for {new_row['date'].iloc[0].date()}.")
    else:
        print("ℹ️ No new data to append.")

if __name__ == "__main__":
    update_data()
