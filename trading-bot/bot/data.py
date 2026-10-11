"""Завантаження свічок (OHLCV): біржа через ccxt (публічні дані, без ключів) або синтетика для тестів."""
from pathlib import Path

import numpy as np
import pandas as pd

CACHE = Path(__file__).resolve().parent.parent / "data"

TF_MS = {"1m": 60_000, "5m": 300_000, "15m": 900_000, "1h": 3_600_000,
         "4h": 14_400_000, "1d": 86_400_000}


def periods_per_year(timeframe: str) -> float:
    return 365 * 86_400_000 / TF_MS[timeframe]


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


def synthetic(n=6000, seed=0, trend=True):
    """Геометричний рандом-волк з режимами (тренд/флет) — лише для перевірки коду."""
    rng = np.random.default_rng(seed)
    regime = np.repeat(rng.choice([-1, 0, 1] if trend else [0], size=n // 300 + 1), 300)[:n]
    ret = regime * 0.0004 + rng.normal(0, 0.006, n)
    close = 100 * np.exp(np.cumsum(ret))
    open_ = np.r_[close[0], close[:-1]]
    spread = np.abs(rng.normal(0, 0.003, n)) * close
    idx = pd.date_range("2024-01-01", periods=n, freq="h")
    return pd.DataFrame({"open": open_, "high": np.maximum(open_, close) + spread,
                         "low": np.minimum(open_, close) - spread, "close": close,
                         "volume": rng.uniform(1, 10, n)}, index=idx)
