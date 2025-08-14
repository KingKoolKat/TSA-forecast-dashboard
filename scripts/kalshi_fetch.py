# scripts/kalshi_fetch.py
import math
import re
from typing import List, Dict, Tuple, Optional

import pandas as pd
import requests


# =========================
# Convenience
# =========================
def week_monday_from_now(ts: Optional[pd.Timestamp] = None) -> pd.Timestamp:
    """Return this week's Monday (normalized to midnight)."""
    now = ts if ts is not None else pd.Timestamp.now()
    return (now.normalize() - pd.to_timedelta(now.weekday(), unit="D"))


# =========================
# Model distribution (from your Monday snapshot)
# =========================
def _norm_cdf(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2)))

def weekly_mu_sigma_from_history(history_df: pd.DataFrame,
                                 week_monday: pd.Timestamp) -> Tuple[float, float]:
    """
    history_df columns required: ['ds','yhat','yhat_lower','yhat_upper','date_made']
    Returns (mu, sigma_week) for the weekly average that was COMMITTED on `week_monday`.
    """
    H = history_df.copy()
    H['ds'] = pd.to_datetime(H['ds']).dt.normalize()
    H['date_made'] = pd.to_datetime(H['date_made']).dt.normalize()
    monday = pd.Timestamp(week_monday).normalize()
    sunday = monday + pd.Timedelta(days=6)

    w = H[(H['date_made'] == monday) & (H['ds'] >= monday) & (H['ds'] <= sunday)].copy()
    if len(w) != 7:
        raise ValueError(
            f"Expected 7 forecast rows for {monday.date()}–{sunday.date()} made on {monday.date()}, found {len(w)}."
        )

    mu = float(w['yhat'].mean())

    # infer per-day sigma from Prophet CI: sigma_i ≈ (upper - lower) / (2*1.96)
    daily_sigma = (w['yhat_upper'] - w['yhat_lower']) / (2 * 1.96)

    # variance of average of 7 days = sum(var_i) / 7^2
    sigma_week = float(math.sqrt((daily_sigma.pow(2).sum()) / (7**2)))
    return mu, sigma_week

def p_above(threshold: float, mu: float, sigma_week: float, safety_scale: float = 1.0) -> float:
    """P(weekly avg > threshold) under Normal(mu, sigma^2)."""
    if sigma_week <= 0:
        return 1.0 if mu > threshold else 0.0
    z = (threshold - mu) / (sigma_week * safety_scale)
    return 1.0 - _norm_cdf(z)


# =========================
# EV math
# =========================
def ev_buy_yes(p_yes: float, price_yes: float, fee_per_share: float = 0.0) -> float:
    # payoff $1 if correct, $0 if wrong, minus fee
    return p_yes * (1 - price_yes) - (1 - p_yes) * price_yes - fee_per_share

def ev_buy_no(p_yes: float, price_no: float, fee_per_share: float = 0.0) -> float:
    # buy NO = win prob is (1 - p_yes)
    p_no = 1 - p_yes
    return p_no * (1 - price_no) - (1 - p_no) * price_no - fee_per_share


# =========================
# Kalshi fetch (public endpoints)
# =========================
BASE = "https://api.elections.kalshi.com/trade-api/v2"

def list_markets(limit: int = 200, cursor: Optional[str] = None) -> Dict:
    params = {"limit": limit}
    if cursor:
        params["cursor"] = cursor
    r = requests.get(f"{BASE}/markets", params=params, timeout=15)
    r.raise_for_status()
    return r.json()

def find_tsa_markets() -> List[Dict]:
    """
    Heuristic filter: titles mentioning TSA / check-ins.
    Adjust filter if Kalshi renames the series.
    """
    out = []
    cursor = None
    while True:
        payload = list_markets(cursor=cursor)
        for m in payload.get("markets", []):
            title = f"{m.get('title','')} {m.get('event_title','')}".lower()
            if "tsa" in title and ("check" in title or "screen" in title or "passenger" in title):
                out.append(m)
        cursor = payload.get("cursor")
        if not cursor:
            break
    return out

def get_orderbook(ticker: str) -> Dict:
    url = f"{BASE}/markets/{ticker}/orderbook"
    r = requests.get(url, timeout=15)
    r.raise_for_status()
    return r.json().get("orderbook", {})

def best_prices_from_orderbook(orderbook: Dict) -> Tuple[Optional[float], Optional[float]]:
    """
    Convert top-of-book bids to approximate asks for YES and NO:
    - YES ask ≈ 1 - best NO bid
    - NO  ask ≈ 1 - best YES bid
    Returns (yes_ask, no_ask) in dollars.
    """
    yes_bids = orderbook.get("yes", [])
    no_bids  = orderbook.get("no", [])
    best_yes_bid = yes_bids[0][0]/100 if yes_bids else None
    best_no_bid  = no_bids[0][0]/100  if no_bids  else None
    yes_ask = (1 - best_no_bid) if best_no_bid is not None else None
    no_ask  = (1 - best_yes_bid) if best_yes_bid is not None else None
    return yes_ask, no_ask

_THR_RE = re.compile(r"above\s+([\d\.,]+)\s*m|above\s+([\d\.,]+)\s*million", re.I)

def parse_threshold_from_title(title: str) -> Optional[float]:
    # Examples: "ABOVE 2.70M", "Above 2.75 million"
    t = title.strip()
    m = _THR_RE.search(t)
    if m:
        num = m.group(1) or m.group(2)
        try:
            return float(num.replace(",", "")) * 1_000_000
        except:
            return None
    # fallback: try to grab a raw number
    m2 = re.search(r"(\d[\d\.,]+)", t)
    if m2:
        try:
            return float(m2.group(1).replace(",", ""))
        except:
            return None
    return None

def fetch_tsa_contracts() -> List[Dict]:
    """
    Returns a list of contracts with fields:
      - label, ticker, threshold, yes_price, no_price
    for TSA 'ABOVE X' markets.
    """
    markets = find_tsa_markets()
    contracts = []
    for m in markets:
        title = m.get("title") or m.get("event_title") or m.get("ticker")
        thr = parse_threshold_from_title(title or "")
        if thr is None:
            continue
        ob = get_orderbook(m["ticker"])
        yes_ask, no_ask = best_prices_from_orderbook(ob)
        contracts.append({
            "label": title,
            "ticker": m["ticker"],
            "threshold": float(thr),
            "yes_price": yes_ask,   # dollars (0..1)
            "no_price":  no_ask,    # dollars (0..1)
        })
    # remove entries where we got neither price
    contracts = [c for c in contracts if (c["yes_price"] is not None or c["no_price"] is not None)]
    return contracts


# =========================
# EV evaluation for ABOVE-X lines (both YES & NO)
# =========================
def evaluate_above_lines(history_csv: str,
                         week_monday: pd.Timestamp,
                         fee_per_share: float = 0.00,
                         safety_scale: float = 1.0) -> pd.DataFrame:
    """
    Loads weekly_forecast_history.csv, computes (mu, sigma) for week_monday,
    fetches TSA ABOVE-X markets & prices, and returns a DataFrame sorted by EV.
    """
    history = pd.read_csv(history_csv, parse_dates=['ds', 'date_made'])
    mu, sigma_week = weekly_mu_sigma_from_history(history, week_monday)

    contracts = fetch_tsa_contracts()

    rows = []
    for c in contracts:
        t = float(c["threshold"])
        p_yes = p_above(t, mu, sigma_week, safety_scale=safety_scale)  # ABOVE => use P(> t)

        yes_p = c.get("yes_price")
        no_p  = c.get("no_price")

        ev_yes_val = ev_buy_yes(p_yes, yes_p, fee_per_share) if yes_p is not None else None
        ev_no_val  = ev_buy_no(p_yes,  no_p,  fee_per_share) if no_p  is not None else None

        best_action, best_ev = None, None
        if ev_yes_val is not None:
            best_action, best_ev = "BUY YES", ev_yes_val
        if ev_no_val is not None and (best_ev is None or ev_no_val > best_ev):
            best_action, best_ev = "BUY NO", ev_no_val

        rows.append({
            "label": c["label"],
            "ticker": c["ticker"],
            "threshold": t,
            "yes_price": yes_p,
            "no_price": no_p,
            "mu_model": mu,
            "sigma_week": sigma_week,
            "p_above_model": p_yes,
            "EV_yes_$": ev_yes_val,
            "EV_no_$": ev_no_val,
            "best_action": best_action,
            "best_EV_$": best_ev,
        })

    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values("best_EV_$", ascending=False)
    return out
