"""Завантаження свічок (OHLCV): біржа через ccxt (публічні дані, без ключів) або синтетика для тестів."""
from pathlib import Path

import numpy as np
import pandas as pd

CACHE = Path(__file__).resolve().parent.parent / "data"

TF_MS = {"1m": 60_000, "5m": 300_000, "15m": 900_000, "1h": 3_600_000,
         "4h": 14_400_000, "1d": 86_400_000}


def periods_per_year(timeframe: str, market: str = "crypto") -> float:
    days = 252 if market == "forex" else 365      # форекс не торгується у вихідні
    return days * 86_400_000 / TF_MS[timeframe]


def fetch_ohlcv(symbol="BTC/USDT", timeframe="1h", limit=5000, exchange="binance", use_cache=True):
    """Тягне історію пагінацією; кешує в data/*.csv."""
    CACHE.mkdir(exist_ok=True)
    path = CACHE / f"{exchange}_{symbol.replace('/', '-')}_{timeframe}.csv"
    if use_cache and path.exists():
        return pd.read_csv(path, index_col=0, parse_dates=True)
    import ccxt
    ex = getattr(ccxt, exchange)({"enableRateLimit": True})
    step = TF_MS[timeframe]
    since = ex.milliseconds() - limit * step
    rows = []
    while len(rows) < limit:
        batch = ex.fetch_ohlcv(symbol, timeframe, since=since, limit=1000)
        if not batch:
            break
        rows += batch
        since = batch[-1][0] + step
        if len(batch) < 2:
            break
    df = _to_df(rows).loc[lambda d: ~d.index.duplicated()]
    df.to_csv(path)
    return df


def fetch_recent(symbol, timeframe, exchange="binance", n=300):
    import ccxt
    ex = getattr(ccxt, exchange)({"enableRateLimit": True})
    return _to_df(ex.fetch_ohlcv(symbol, timeframe, limit=n))


def _to_df(rows):
    df = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close", "volume"])
    df.index = pd.to_datetime(df.pop("ts"), unit="ms")
    return df.astype(float)


def synthetic(n=6000, seed=0, trend=True, start=100.0, vol=0.006):
    """Геометричний рандом-волк з режимами (тренд/флет) — лише для перевірки коду."""
    rng = np.random.default_rng(seed)
    regime = np.repeat(rng.choice([-1, 0, 1] if trend else [0], size=n // 300 + 1), 300)[:n]
    ret = regime * (vol / 15) + rng.normal(0, vol, n)
    close = start * np.exp(np.cumsum(ret))
    open_ = np.r_[close[0], close[:-1]]
    spread = np.abs(rng.normal(0, vol / 2, n)) * close
    idx = pd.date_range("2024-01-01", periods=n, freq="h")
    return pd.DataFrame({"open": open_, "high": np.maximum(open_, close) + spread,
                         "low": np.minimum(open_, close) - spread, "close": close,
                         "volume": rng.uniform(1, 10, n)}, index=idx)


# ---------- Forex ----------
OANDA_TF = {"5m": "M5", "15m": "M15", "1h": "H1", "4h": "H4", "1d": "D"}


def load_csv(path):
    """CSV з Dukascopy / MT5 / TradingView: шукає колонку часу й OHLC за назвами."""
    df = pd.read_csv(path, sep=None, engine="python")
    df.columns = [str(c).strip().lower().replace("<", "").replace(">", "") for c in df.columns]
    tcol = next((c for c in df.columns if c in ("time", "date", "datetime", "gmt time", "timestamp", "local time")), None)
    if tcol is None and {"date", "time"} <= set(df.columns):
        pass
    if "date" in df.columns and "time" in df.columns:
        idx = pd.to_datetime(df["date"].astype(str) + " " + df["time"].astype(str))
    else:
        idx = pd.to_datetime(df[tcol], dayfirst=True, utc=True).dt.tz_localize(None)
    out = df[["open", "high", "low", "close"]].astype(float)
    out["volume"] = df["volume"].astype(float) if "volume" in df.columns else 0.0
    out.index = idx
    return out.sort_index().loc[lambda d: ~d.index.duplicated()]


def fetch_forex(instrument="EUR_USD", timeframe="1h", bars=20000, source="oanda", csv=None):
    """source: csv (файл), oanda (потрібен OANDA_TOKEN, практика безкоштовна), yahoo (≤730 днів 1h)."""
    if csv:
        return load_csv(csv)
    CACHE.mkdir(exist_ok=True)
    path = CACHE / f"{source}_{instrument}_{timeframe}.csv"
    if path.exists():
        return pd.read_csv(path, index_col=0, parse_dates=True)
    if source == "oanda":
        df = _oanda_candles(instrument, timeframe, bars)
    else:
        import yfinance as yf
        d = yf.download(instrument.replace("_", "") + "=X", period="730d", interval=timeframe,
                        progress=False, auto_adjust=False)
        d.columns = [c[0].lower() for c in d.columns]
        df = d[["open", "high", "low", "close", "volume"]].astype(float)
        df.index = df.index.tz_localize(None)
    df.to_csv(path)
    return df


def _oanda_candles(instrument, timeframe, bars, last_n=None):
    import os
    import requests
    tok = os.environ.get("OANDA_TOKEN")
    if not tok:
        raise SystemExit("Задай OANDA_TOKEN (practice-токен з oanda.com, безкоштовний демо-рахунок)")
    host = "api-fxtrade.oanda.com" if os.environ.get("OANDA_LIVE") == "1" else "api-fxpractice.oanda.com"
    hdr = {"Authorization": f"Bearer {tok}"}
    gran, rows = OANDA_TF[timeframe], []
    if last_n:
        r = requests.get(f"https://{host}/v3/instruments/{instrument}/candles", headers=hdr,
                         params={"granularity": gran, "count": last_n, "price": "M"}, timeout=20)
        r.raise_for_status(); cands = r.json()["candles"]
        return _oanda_df([c for c in cands if c["complete"]])
    frm = pd.Timestamp.utcnow().tz_localize(None) - pd.Timedelta(milliseconds=bars * TF_MS[timeframe] * 7 / 5)
    while len(rows) < bars:
        r = requests.get(f"https://{host}/v3/instruments/{instrument}/candles", headers=hdr, timeout=30,
                         params={"granularity": gran, "from": frm.strftime("%Y-%m-%dT%H:%M:%S.000000000Z"),
                                 "count": 5000, "price": "M"})
        r.raise_for_status()
        cands = [c for c in r.json()["candles"] if c["complete"]]
        if not cands:
            break
        rows += cands
        frm = pd.Timestamp(cands[-1]["time"]).tz_localize(None) + pd.Timedelta(seconds=1)
    return _oanda_df(rows)


def _oanda_df(cands):
    df = pd.DataFrame({"open": [float(c["mid"]["o"]) for c in cands], "high": [float(c["mid"]["h"]) for c in cands],
                       "low": [float(c["mid"]["l"]) for c in cands], "close": [float(c["mid"]["c"]) for c in cands],
                       "volume": [c["volume"] for c in cands]},
                      index=pd.to_datetime([c["time"] for c in cands]).tz_localize(None))
    return df.loc[~df.index.duplicated()]
