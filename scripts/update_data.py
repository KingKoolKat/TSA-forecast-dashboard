# scripts/update_data.py

import requests
import pandas as pd
from bs4 import BeautifulSoup
from datetime import datetime

# === CONFIG ===
TSA_URL = "https://www.tsa.gov/travel/passenger-volumes"
DATA_PATH = "data/tsa_daily_full.csv"

def fetch_latest_tsa_rows():
    """Fetch full TSA table from website."""
    resp = requests.get(TSA_URL)
    soup = BeautifulSoup(resp.text, "lxml")
    table = soup.find("table")
    df = pd.read_html(str(table))[0]

    # Standardize columns
    df = df.rename(columns={
        'Date': 'date',
        'Numbers': 'throughput'
    })
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')
    return df

def load_existing_data():
    return pd.read_csv(DATA_PATH, parse_dates=['date'])

def update_data():
    latest_df = fetch_latest_tsa_rows()
    existing = load_existing_data()

    last_date = existing['date'].max()
    new_rows = latest_df[latest_df['date'] > last_date]

    if not new_rows.empty:
        updated = pd.concat([existing, new_rows], ignore_index=True)
        updated.to_csv(DATA_PATH, index=False)
        print(f"✅ Added {len(new_rows)} new rows: {', '.join(d.strftime('%Y-%m-%d') for d in new_rows['date'])}")
    else:
        print("ℹ️ No new data to append.")

if __name__ == "__main__":
    update_data()
