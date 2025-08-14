# scripts/kalshi_fetch.py
import re
import math
import requests
import pandas as pd
from typing import Optional, Tuple, List, Dict

# Try both bases; some envs expose one or the other
BASES = [
    "https://api.elections.kalshi.com/trade-api/v2",
    "https://api.kalshi.com/trade-api/v2",
]

def _get(url_path: str, **kwargs) -> requests.Response:
    last_err = None
    for base in BASES:
        try:
            r = requests.get(f"{base}{url_path}", timeout=15, **kwargs)
            r.raise_for_status()
            return r
        except Exception as e:
            last_err = e
            continue
    raise RuntimeError(f"All Kalshi endpoints failed for {url_path}: {last_err}")

def list_markets(limit: int = 200, cursor: Optional[str] = None) -> Dict:
    params = {"limit": limit}
    if cursor:
        params["cursor"] = cursor
    return _get("/markets", params=params).json()

def get_orderbook(ticker: str) -> Dict:
    return _get(f"/markets/{ticker}/orderbook").json().get("orderbook", {})

# looser TSA filter
def _looks_like_tsa(title: str) -> bool:
    t = (title or "").lower()
    return ("tsa" in t) or ("screen" in t) or ("check" in t) or ("passenger" in t)

def find_tsa_markets() -> List[Dict]:
    out = []
    cursor = None
    while True:
        payload = list_markets(cursor=cursor)
        for m in payload.get("markets", []):
            title = f"{m.get('title','')} {m.get('event_title','')}".strip()
            if _looks_like_tsa(title):
                out.append(m)
        cursor = payload.get("cursor")
        if not cursor:
            break
    return out

# parse thresholds like "ABOVE 2.70M", "Above 2.75 million", "Above 2,700,000"
_THR_RE = re.compile(r"above\s+([\d\.,]+)\s*m(?:illion)?", re.I)

def parse_threshold_from_title(title: str) -> Optional[float]:
    t = title or ""
    m = _THR_RE.search(t)
    if m:
        return float(m.group(1).replace(",", "")) * 1_000_000
    # fallback to a big number in title
    m2 = re.search(r"(\d[\d,\.]+)", t)
    if m2:
        try:
            val = float(m2.group(1).replace(",", ""))
            # assume already in raw passengers if > 10^6
            return val if val > 1e6 else val * 1_000_000
        except:
            return None
    return None

def best_prices_from_orderbook(orderbook: Dict) -> Tuple[Optional[float], Optional[float]]:
    yes_bids = orderbook.get("yes", [])
    no_bids  = orderbook.get("no", [])
    best_yes_bid = yes_bids[0][0]/100 if yes_bids else None
    best_no_bid  = no_bids[0][0]/100  if no_bids  else None
    # Derive asks from opposite bids if possible
    yes_ask = (1 - best_no_bid) if best_no_bid is not None else None
    no_ask  = (1 - best_yes_bid) if best_yes_bid is not None else None
    return yes_ask, no_ask

def fetch_tsa_contracts() -> List[Dict]:
    """Return contracts with possible None prices; do NOT drop rows."""
    markets = find_tsa_markets()
    contracts = []
    for m in markets:
        title  = m.get("title") or m.get("event_title") or m.get("ticker")
        thr = parse_threshold_from_title(title)
        if thr is None:
            continue
        try:
            ob = get_orderbook(m["ticker"])
            yes_ask, no_ask = best_prices_from_orderbook(ob)
        except Exception:
            yes_ask, no_ask = None, None
        contracts.append({
            "label": title,
            "ticker": m.get("ticker"),
            "threshold": float(thr),
            "yes_price": yes_ask,   # 0..1 or None
            "no_price":  no_ask,    # 0..1 or None
        })
    return contracts

# ==== model bits unchanged (you should already have these) ====
def _norm_cdf(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2)))

def weekly_mu_sigma_from_history(history_df: pd.DataFrame, week_monday: pd.Timestamp) -> Tuple[float, float]:
    H = history_df.copy()
    H['ds'] = pd.to_datetime(H['ds']).dt.normalize()
    H['date_made'] = pd.to_datetime(H['date_made']).dt.normalize()
    monday = pd.Timestamp(week_monday).normalize()
    sunday = monday + pd.Timedelta(days=6)
    wk = H[(H['date_made'] == monday) & (H['ds'] >= monday) & (H['ds'] <= sunday)].copy()
    if len(wk) != 7:
        raise ValueError(f"Expected 7 rows for {monday.date()}–{sunday.date()}, found {len(wk)}.")
    mu = float(wk['yhat'].mean())
    daily_sigma = (wk['yhat_upper'] - wk['yhat_lower']) / (2 * 1.96)
    sigma_week = float(((daily_sigma.pow(2).sum()) / (7**2)) ** 0.5)
    return mu, sigma_week

def p_above(threshold: float, mu: float, sigma_week: float, safety_scale: float = 1.0) -> float:
    if sigma_week <= 0:
        return 1.0 if mu > threshold else 0.0
    z = (threshold - mu) / (sigma_week * safety_scale)
    return 1.0 - _norm_cdf(z)

def ev_buy_yes(p_yes: float, price_yes: float, fee_per_share: float = 0.0) -> float:
    return p_yes * (1 - price_yes) - (1 - p_yes) * price_yes - fee_per_share

def ev_buy_no(p_yes: float, price_no: float, fee_per_share: float = 0.0) -> float:
    p_no = 1 - p_yes
    return p_no * (1 - price_no) - (1 - p_no) * price_no - fee_per_share

def evaluate_above_lines(history_csv: str,
                         week_monday: pd.Timestamp,
                         fee_per_share: float = 0.00,
                         safety_scale: float = 1.0) -> pd.DataFrame:
    history = pd.read_csv(history_csv, parse_dates=['ds','date_made'])
    mu, sigma_week = weekly_mu_sigma_from_history(history, week_monday)
    contracts = fetch_tsa_contracts()

    rows = []
    for c in contracts:
        t = float(c["threshold"])
        p_yes = p_above(t, mu, sigma_week, safety_scale=safety_scale)
        yes_p = c.get("yes_price")
        no_p  = c.get("no_price")
        ev_yes_val = ev_buy_yes(p_yes, yes_p, fee_per_share) if yes_p is not None else None
        ev_no_val  = ev_buy_no(p_yes,  no_p,  fee_per_share) if no_p  is not None else None

        # choose best if at least one price exists
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
