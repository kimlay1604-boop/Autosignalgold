"""
Signal bot for XAU/USD (gold) and BTC/USD (Bitcoin).

Strategy (simple, rule-based, for learning/testing):
  BUY  when EMA20 crosses above EMA50 and RSI14 > 50
  SELL when EMA20 crosses below EMA50 and RSI14 < 50

  XAU/USD: stop 10.00 USD, target 15.00 USD (1:1.5), 1 pip = 0.10
  BTC/USD: stop = 2 x ATR, target = 3 x ATR (1:1.5), so it adapts to Bitcoin's size.
           NOTE: BTC settings are NOT backtested yet. Treat them as a starting point.

Every signal is tracked against later candles and marked
OPEN / WIN / SL HIT honestly. If stop and target are both touched in
the same candle, it is counted as a loss (conservative).

Usage:
  pip install pandas numpy yfinance
  python bot.py --demo                    # fake random data, works offline
  python bot.py                           # live data for gold and Bitcoin, loops every 60s
  python bot.py --symbols "XAU/USD"       # only gold
  python bot.py --once                    # run a single time and exit
  python bot.py --ai                      # Claude analyzes the market (needs ANTHROPIC_API_KEY)

Then serve the folder:  python -m http.server 8000
and open http://localhost:8000
"""
import argparse
import json
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

INTERVAL = "15m"
SYMBOLS = {
    "XAU/USD": dict(ticker="GC=F", prefix="", stop=10.0, target=15.0, pip=0.10,
                    base=4150, vol=2.2, wick=2.5, name="gold (XAU/USD)"),
    "BTC/USD": dict(ticker="BTC-USD", prefix="BTC-", atr_stop=2.0, atr_target=3.0, pip=1.0,
                    base=90000, vol=120, wick=135, name="Bitcoin (BTC/USD)"),
}

# kept for autotrader.py (gold only)
SYMBOL_LABEL = "XAU/USD"
STOP_USD = SYMBOLS[SYMBOL_LABEL]["stop"]
TARGET_USD = SYMBOLS[SYMBOL_LABEL]["target"]
PIP = SYMBOLS[SYMBOL_LABEL]["pip"]

OUT = Path(os.environ.get("SIGNALS_OUT") or Path(__file__).with_name("signals.json"))
MAX_KEEP = 60                # per symbol


def get_live_data(symbol: str = SYMBOL_LABEL) -> pd.DataFrame:
    import yfinance as yf
    df = yf.download(SYMBOLS[symbol]["ticker"], period="5d", interval=INTERVAL, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.rename(columns=str.lower).dropna()
    df.index = pd.to_datetime(df.index, utc=True)
    return df[["open", "high", "low", "close"]]


def get_demo_data(n: int = 700, seed: int | None = None, symbol: str = SYMBOL_LABEL) -> pd.DataFrame:
    cfg = SYMBOLS[symbol]
    rng = np.random.default_rng(seed)
    close = cfg["base"] + np.cumsum(rng.normal(0, cfg["vol"], n))
    open_ = np.r_[close[0], close[:-1]]
    high = np.maximum(open_, close) + rng.uniform(0, cfg["wick"], n)
    low = np.minimum(open_, close) - rng.uniform(0, cfg["wick"], n)
    idx = pd.date_range(end=pd.Timestamp.now(tz="UTC").floor("15min"), periods=n, freq="15min")
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close}, index=idx)


def rsi(series: pd.Series, n: int = 14) -> pd.Series:
    d = series.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn.replace(0, np.nan))


def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    c = df["close"]
    tr = pd.concat([df["high"] - df["low"], (df["high"] - c.shift()).abs(),
                    (df["low"] - c.shift()).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False).mean()


def find_signals(df: pd.DataFrame, symbol: str = SYMBOL_LABEL) -> list[dict]:
    cfg = SYMBOLS[symbol]
    df = df.copy()
    df["ema_fast"] = df["close"].ewm(span=20, adjust=False).mean()
    df["ema_slow"] = df["close"].ewm(span=50, adjust=False).mean()
    df["rsi"] = rsi(df["close"])
    df["atr"] = atr(df)
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
        stop_d = cfg["stop"] if "stop" in cfg else round(float(row["atr"]) * cfg["atr_stop"], 2)
        target_d = cfg["target"] if "target" in cfg else round(float(row["atr"]) * cfg["atr_target"], 2)
        entry = round(float(row["close"]), 2)
        sign = 1 if side == "BUY" else -1
        signals.append({
            "id": f"{cfg['prefix']}{ts.strftime('%Y%m%d%H%M')}",
            "symbol": symbol,
            "side": side,
            "time": ts.isoformat(),
            "entry": entry,
            "stop": round(entry - sign * stop_d, 2),
            "target": round(entry + sign * target_d, 2),
            "stop_pips": int(round(stop_d / cfg["pip"])),
            "target_pips": int(round(target_d / cfg["pip"])),
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


def ai_review(df: pd.DataFrame, candidate: dict | None = None, symbol: str | None = None) -> dict:
    """Ask Claude to analyze the market. Needs ANTHROPIC_API_KEY.
    Returns {"verdict": "BUY"|"SELL"|"WAIT", "confidence": 0-100, "reason": str}."""
    import anthropic

    symbol = symbol or (candidate or {}).get("symbol") or SYMBOL_LABEL
    d = df.copy()
    c = d["close"]
    d["ema20"] = c.ewm(span=20, adjust=False).mean()
    d["ema50"] = c.ewm(span=50, adjust=False).mean()
    d["ema200"] = c.ewm(span=200, adjust=False).mean()
    d["rsi14"] = rsi(c)
    d["atr14"] = atr(d)
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
        f"You are a careful {SYMBOLS[symbol]['name']} technical analyst. Below are the latest 15-minute candles "
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


def _tally(sigs: list[dict]) -> dict:
    w = sum(s["status"] == "WIN" for s in sigs)
    l = sum(s["status"] == "SL HIT" for s in sigs)
    return {"wins": w, "losses": l, "open": len(sigs) - w - l}


def run_once(demo: bool, use_ai: bool = False, symbols: list[str] | None = None) -> None:
    symbols = symbols or list(SYMBOLS)
    prev = json.loads(OUT.read_text()) if OUT.exists() else {}
    old = {s["id"]: s for s in prev.get("signals", [])}
    rejected = set(prev.get("rejected", []))

    views = prev.get("market_view") or {}
    if "verdict" in views:                       # older file format: one view for gold
        views = {SYMBOL_LABEL: views}
    view_bars = prev.get("view_bar") or {}
    if isinstance(view_bars, str):
        view_bars = {SYMBOL_LABEL: view_bars}

    merged = dict(old)
    dfs, prices = {}, dict(prev.get("prices") or {})
    for sym in symbols:
        try:
            df = get_demo_data(symbol=sym) if demo else get_live_data(sym)
        except Exception as e:                   # one symbol failing must not stop the other
            print(f"{sym}: could not get data ({e})")
            continue
        dfs[sym] = df
        prices[sym] = round(float(df["close"].iloc[-1]), 2)
        last_bar = df.index[-1].isoformat()

        if use_ai and view_bars.get(sym) != last_bar:      # one market read per new candle
            try:
                views[sym], view_bars[sym] = ai_review(df, None, sym), last_bar
            except Exception as e:
                print(f"{sym}: AI market view failed:", e)

        cutoff = df.index[-REVIEW_RECENT_BARS]
        for s in find_signals(df, sym):
            if s["id"] in merged or s["id"] in rejected:
                continue
            if use_ai and pd.Timestamp(s["time"]) >= cutoff:   # brand-new signal: ask the AI
                try:
                    r = ai_review(df, s, sym)
                except Exception as e:
                    print(f"{sym}: AI review failed, skipping signal:", e)
                    continue
                if r["verdict"] != s["side"] or r["confidence"] < MIN_CONFIDENCE:
                    rejected.add(s["id"])
                    print(f"{sym}: AI rejected {s['side']} @ {s['entry']}: {r['reason']}")
                    continue
                s["ai"] = r
            merged[s["id"]] = s

    sigs = []
    for sym in SYMBOLS:
        mine = sorted((s for s in merged.values() if s["symbol"] == sym),
                      key=lambda s: s["time"], reverse=True)[:MAX_KEEP]
        if sym in dfs:
            resolve(mine, dfs[sym])
        sigs += mine
    sigs.sort(key=lambda s: s["time"], reverse=True)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "updated": datetime.now(timezone.utc).isoformat(),
        "demo": demo,
        "ai": use_ai,
        "last_price": prices.get(SYMBOL_LABEL),
        "prices": prices,
        "market_view": views,
        "view_bar": view_bars,
        "rejected": sorted(rejected)[-200:],
        "stats": _tally(sigs),
        "stats_by_symbol": {sym: _tally([s for s in sigs if s["symbol"] == sym]) for sym in SYMBOLS},
        "signals": sigs,
    }, indent=2))
    t = _tally(sigs)
    print(f"{datetime.now():%H:%M:%S}  signals={len(sigs)}  W={t['wins']} L={t['losses']}  prices={prices}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", action="store_true", help="use fake random data")
    ap.add_argument("--once", action="store_true", help="run once and exit")
    ap.add_argument("--ai", action="store_true", help="let Claude analyze the market and approve signals")
    ap.add_argument("--symbols", default=",".join(SYMBOLS), help='comma separated, e.g. "XAU/USD,BTC/USD"')
    a = ap.parse_args()
    chosen = [x.strip() for x in a.symbols.split(",") if x.strip()]
    bad = [x for x in chosen if x not in SYMBOLS]
    if bad:
        raise SystemExit(f"Unknown symbol(s) {bad}. Choose from {list(SYMBOLS)}")
    while True:
        try:
            run_once(a.demo, a.ai, chosen)
        except Exception as e:
            print("error:", e)
        if a.once:
            break
        time.sleep(60)
