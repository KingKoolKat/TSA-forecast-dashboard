# scripts/tune_prophet.py
# Tune Prophet for next-week TSA forecasts with rolling CV + resumable CSV logging.

import os, csv, itertools, time, random
from pathlib import Path

import numpy as np
import pandas as pd

random.seed(42); np.random.seed(42)

try:
    from prophet import Prophet
    from prophet.diagnostics import cross_validation
except Exception:
    from fbprophet import Prophet
    from fbprophet.diagnostics import cross_validation

DATA_CSV = Path("data/tsa_daily_full.csv")

def load_df(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["date"])
    df = df.rename(columns={"date": "ds", "throughput": "y"}).sort_values("ds")
    return df

def weekly_mape_from_cv(df_cv: pd.DataFrame) -> float:
    out = df_cv.copy()
    if "horizon" not in out.columns:
        out["horizon"] = pd.to_datetime(out["ds"]) - pd.to_datetime(out["cutoff"])
    out7 = out[out["horizon"] <= pd.Timedelta("7 days")].copy()
    weekly = (
        out7.groupby("cutoff", as_index=False)
            .agg(y_week=("y", "mean"), yhat_week=("yhat", "mean"))
    )
    weekly["ape"] = (weekly["yhat_week"] - weekly["y_week"]).abs() / weekly["y_week"].clip(lower=1e-9)
    return float(weekly["ape"].mean() * 100.0)

GRID = {
    "changepoint_prior_scale": [0.01, 0.05, 0.1, 0.5],
    "seasonality_mode": ["multiplicative", "additive"],
    "seasonality_prior_scale": [1, 5, 10],
    "holidays_prior_scale": [1, 5, 10],
    "changepoint_range": [0.8, 0.9, 0.95, 1.0],
}

INITIAL = "730 days"
HORIZON = "7 days"
PERIOD  = "7 days"

RESULTS_CSV = Path("tuning_results.csv")

def try_fit_and_score(df: pd.DataFrame, cfg: tuple) -> float:
    cps, smode, sps, hps, crange = cfg
    m = Prophet(
        changepoint_prior_scale=cps,
        seasonality_mode=smode,
        seasonality_prior_scale=sps,
        holidays_prior_scale=hps,
        changepoint_range=crange,
        yearly_seasonality=True,
        weekly_seasonality=True,
    )
    m.add_country_holidays(country_name="US")
    m.fit(df)
    df_cv = cross_validation(m, initial=INITIAL, horizon=HORIZON, period=PERIOD)
    wk_mape = weekly_mape_from_cv(df_cv)
    del df_cv
    return wk_mape

def main():
    df = load_df(DATA_CSV)
    all_cfgs = list(itertools.product(
        GRID["changepoint_prior_scale"],
        GRID["seasonality_mode"],
        GRID["seasonality_prior_scale"],
        GRID["holidays_prior_scale"],
        GRID["changepoint_range"],
    ))
    total = len(all_cfgs)

    done_keys = set()
    if RESULTS_CSV.exists():
        df_done = pd.read_csv(RESULTS_CSV)
        for _, r in df_done.iterrows():
            key = (r["changepoint_prior_scale"], r["seasonality_mode"],
                   r["seasonality_prior_scale"], r["holidays_prior_scale"],
                   r["changepoint_range"])
            done_keys.add(key)
    else:
        with open(RESULTS_CSV, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow([
                "changepoint_prior_scale","seasonality_mode",
                "seasonality_prior_scale","holidays_prior_scale",
                "changepoint_range","weekly_mape","status","notes"
            ])

    start = time.time()
    for i, cfg in enumerate(all_cfgs, 1):
        key = tuple(cfg)
        if key in done_keys:
            print(f"[{i}/{total}] SKIP {key}")
            continue
        print(f"[{i}/{total}] RUN  {key}")
        status, notes = "ok", ""
        try:
            wk_mape = try_fit_and_score(df, cfg)
        except Exception as e:
            wk_mape, status, notes = float("nan"), "fail", repr(e)
        with open(RESULTS_CSV, "a", newline="") as f:
            w = csv.writer(f)
            w.writerow([cfg[0], cfg[1], cfg[2], cfg[3], cfg[4], wk_mape, status, notes])
            f.flush(); os.fsync(f.fileno())
        elapsed = int(time.time() - start)
        print(f"   -> weekly_mape={wk_mape:.3f} ({status}), elapsed={elapsed}s")

    res = pd.read_csv(RESULTS_CSV)
    res_ok = res[res["status"]=="ok"].sort_values("weekly_mape", na_position="last")
    print("\nTop 10 by next-week MAPE:")
    print(res_ok.head(10).to_string(index=False))

if __name__ == "__main__":
    main()
