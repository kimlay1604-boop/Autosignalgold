"""
XAU/USD signal bot.

Strategy (simple, rule-based, for learning/testing):
  BUY  when EMA20 crosses above EMA50 and RSI14 > 50
  SELL when EMA20 crosses below EMA50 and RSI14 < 50
  Stop   = 10.00 USD from entry  (shown as 100 pips, 1 pip = 0.10)
  Target = 15.00 USD from entry  (shown as +150 pips, 1:1.5 risk/reward)

Every signal is tracked against later candles and marked
OPEN / WIN / SL HIT honestly. If stop and target are both touched in
the same candle, it is counted as a loss (conservative).

Usage:
  pip install pandas numpy yfinance
  python bot.py --demo          # fake random data, works offline
  python bot.py                 # live gold data (yfinance, GC=F), loops every 60s
  python bot.py --once          # run a single time and exit

Then serve the folder:  python -m http.server 8000
and open http://localhost:8000
"""
import argparse
import os
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

SYMBOL_LABEL = "XAU/USD"
YF_TICKER = "GC=F"          # gold futures; close to spot XAU/USD
INTERVAL = "15m"
STOP_USD = 10.0             # 100 pips
TARGET_USD = 15.0           # 150 pips
PIP = 0.10
OUT = Path(os.environ.get("SIGNALS_OUT") or Path(__file__).with_name("signals.json"))
MAX_KEEP = 60


def get_live_data() -> pd.DataFrame:
    import yfinance as yf
    df = yf.download(YF_TICKER, period="5d", interval=INTERVAL, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.rename(columns=str.lower).dropna()
    df.index = pd.to_datetime(df.index, utc=True)
    return df[["open", "high", "low", "close"]]


def get_demo_data(n: int = 700, seed: int | None = None) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = 4150 + np.cumsum(rng.normal(0, 2.2, n))
    open_ = np.r_[close[0], close[:-1]]
    high = np.maximum(open_, close) + rng.uniform(0, 2.5, n)
    low = np.minimum(open_, close) - rng.uniform(0, 2.5, n)
    idx = pd.date_range(end=pd.Timestamp.now(tz="UTC").floor("15min"), periods=n, freq="15min")
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close}, index=idx)


def rsi(series: pd.Series, n: int = 14) -> pd.Series:
    d = series.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn.replace(0, np.nan))


def find_signals(df: pd.DataFrame) -> list[dict]:
    df = df.copy()
    df["ema_fast"] = df["close"].ewm(span=20, adjust=False).mean()
    df["ema_slow"] = df["close"].ewm(span=50, adjust=False).mean()
    df["rsi"] = rsi(df["close"])
    above = df["ema_fast"] > df["ema_slow"]
    cross_up = above & ~above.shift(1, fill_value=False)
    cross_dn = ~above & above.shift(1, fill_value=True)

    signals = []
    for ts, row in df.iloc[60:].iterrows():
        side = None
        if cross_up[ts] and row["rsi"] > 50:
            side = "BUY"
        elif cross_dn[ts] and row["rsi"] < 50:
            side = "SELL"
        if not side:
            continue
        entry = round(float(row["close"]), 2)
        sign = 1 if side == "BUY" else -1
        signals.append({
            "id": ts.strftime("%Y%m%d%H%M"),
            "symbol": SYMBOL_LABEL,
            "side": side,
            "time": ts.isoformat(),
            "entry": entry,
            "stop": round(entry - sign * STOP_USD, 2),
            "target": round(entry + sign * TARGET_USD, 2),
            "stop_pips": int(STOP_USD / PIP),
            "target_pips": int(TARGET_USD / PIP),
            "status": "OPEN",
            "closed_at": None,
        })
    return signals


def resolve(signals: list[dict], df: pd.DataFrame) -> None:
    for s in signals:
        if s["status"] != "OPEN":
            continue
        later = df[df.index > pd.Timestamp(s["time"])]
        for ts, c in later.iterrows():
            if s["side"] == "BUY":
                hit_sl, hit_tp = c["low"] <= s["stop"], c["high"] >= s["target"]
            else:
                hit_sl, hit_tp = c["high"] >= s["stop"], c["low"] <= s["target"]
            if hit_sl:                      # SL wins ties (conservative)
                s["status"], s["closed_at"] = "SL HIT", ts.isoformat()
                break
            if hit_tp:
                s["status"], s["closed_at"] = "WIN", ts.isoformat()
                break


AI_MODEL = "claude-sonnet-5-5"
MIN_CONFIDENCE = 60          # AI must be at least this sure to approve a signal
REVIEW_RECENT_BARS = 2       # only AI-review signals from the last N candles


def ai_review(df: pd.DataFrame, candidate: dict | None = None) -> dict:
    """Ask Claude to analyze the market. Needs ANTHROPIC_API_KEY.
    Returns {"verdict": "BUY"|"SELL"|"WAIT", "confidence": 0-100, "reason": str}."""
    import re
    import anthropic

    d = df.copy()
    c = d["close"]
    d["ema20"] = c.ewm(span=20, adjust=False).mean()
    d["ema50"] = c.ewm(span=50, adjust=False).mean()
    d["ema200"] = c.ewm(span=200, adjust=False).mean()
    d["rsi14"] = rsi(c)
    tr = pd.concat([d["high"] - d["low"], (d["high"] - c.shift()).abs(), (d["low"] - c.shift()).abs()], axis=1).max(axis=1)
    d["atr14"] = tr.rolling(14).mean()
    table = d.tail(48).round(2).to_csv()

    task = (
        f"A rule-based system proposes a {candidate['side']} at {candidate['entry']} "
        f"with stop {candidate['stop']} and target {candidate['target']}. "
        "Decide whether to APPROVE it (answer with the same side) or reject it (answer WAIT). "
        "Consider trend (EMA20/50/200), momentum (RSI), volatility (ATR vs the stop distance), "
        "nearby swing highs/lows that could block the target, and whether price is overextended."
        if candidate else
        "Describe the current market bias and say whether a BUY, SELL or WAIT setup looks best right now."
    )
    prompt = (
        "You are a careful gold (XAU/USD) technical analyst. Below are the latest 15-minute candles "
        "with indicators (oldest first).\n\n" + table + "\n\n" + task +
        "\n\nBe skeptical: most moments deserve WAIT. Reply with ONLY a JSON object: "
        '{"verdict": "BUY" | "SELL" | "WAIT", "confidence": <0-100>, "reason": "<max 2 short sentences>"}'
    )
    msg = anthropic.Anthropic().messages.create(
        model=AI_MODEL, max_tokens=300, messages=[{"role": "user", "content": prompt}]
    )
    text = "".join(b.text for b in msg.content if b.type == "text")
    out = json.loads(re.search(r"\{.*\}", text, re.S).group(0))
    out["verdict"] = str(out.get("verdict", "WAIT")).upper()
    out["confidence"] = int(out.get("confidence", 0))
    return out


def run_once(demo: bool, use_ai: bool = False) -> None:
    df = get_demo_data() if demo else get_live_data()
    prev = json.loads(OUT.read_text()) if OUT.exists() else {}
    old = {s["id"]: s for s in prev.get("signals", [])}
    rejected = set(prev.get("rejected", []))
    market_view, view_bar = prev.get("market_view"), prev.get("view_bar")
    last_bar = df.index[-1].isoformat()

    if use_ai and view_bar != last_bar:               # one market read per new candle
        try:
            market_view, view_bar = ai_review(df), last_bar
        except Exception as e:
            print("AI market view failed:", e)

    cutoff = df.index[-REVIEW_RECENT_BARS]
    merged = dict(old)
    for s in find_signals(df):
        if s["id"] in merged or s["id"] in rejected:
            continue
        if use_ai and pd.Timestamp(s["time"]) >= cutoff:   # brand-new signal: ask the AI
            try:
                r = ai_review(df, s)
            except Exception as e:
                print("AI review failed, skipping signal:", e)
                continue
            if r["verdict"] != s["side"] or r["confidence"] < MIN_CONFIDENCE:
                rejected.add(s["id"])
                print(f"AI rejected {s['side']} @ {s['entry']}: {r['reason']}")
                continue
            s["ai"] = r
        merged[s["id"]] = s

    sigs = sorted(merged.values(), key=lambda s: s["time"], reverse=True)[:MAX_KEEP]
    resolve(sigs, df)
    wins = sum(s["status"] == "WIN" for s in sigs)
    losses = sum(s["status"] == "SL HIT" for s in sigs)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "updated": datetime.now(timezone.utc).isoformat(),
        "demo": demo,
        "ai": use_ai,
        "last_price": round(float(df["close"].iloc[-1]), 2),
        "market_view": market_view,
        "view_bar": view_bar,
        "rejected": sorted(rejected)[-200:],
        "stats": {"wins": wins, "losses": losses, "open": len(sigs) - wins - losses},
        "signals": sigs,
    }, indent=2))
    print(f"{datetime.now():%H:%M:%S}  signals={len(sigs)}  W={wins} L={losses}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", action="store_true", help="use fake random data")
    ap.add_argument("--once", action="store_true", help="run once and exit")
    ap.add_argument("--ai", action="store_true", help="let Claude analyze the market and approve signals")
    a = ap.parse_args()
    while True:
        try:
            run_once(a.demo, a.ai)
        except Exception as e:
            print("error:", e)
        if a.once:
            break
        time.sleep(60)
