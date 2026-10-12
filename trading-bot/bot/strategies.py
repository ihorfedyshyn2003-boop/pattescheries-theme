"""Стратегії: функція (df, **params) -> позиція {0,1} (long/flat) на кожній свічці.
Позиція зсувається на 1 бар у бектесті — заглядання в майбутнє виключене."""
import numpy as np
import pandas as pd


def sma_cross(df, fast=20, slow=50):
    return (df.close.rolling(fast).mean() > df.close.rolling(slow).mean()).astype(float)


def rsi(close, n):
    d = close.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn.replace(0, np.nan))


def rsi_reversion(df, n=14, buy=30, sell=55):
    r = rsi(df.close, n)
    pos = pd.Series(np.nan, index=df.index)
    pos[r < buy] = 1.0
    pos[r > sell] = 0.0
    return pos.ffill().fillna(0.0)


def donchian(df, entry=55, exit=20):
    up = df.high.rolling(entry).max().shift(1)
    lo = df.low.rolling(exit).min().shift(1)
    pos = pd.Series(np.nan, index=df.index)
    pos[df.close > up] = 1.0
    pos[df.close < lo] = 0.0
    return pos.ffill().fillna(0.0)


def trend_filter_pullback(df, trend=200, n=5, buy=25):
    """Купуємо просідання RSI лише в аптренді (ціна > SMA trend)."""
    up = df.close > df.close.rolling(trend).mean()
    r = rsi(df.close, n)
    pos = pd.Series(np.nan, index=df.index)
    pos[(r < buy) & up] = 1.0
    pos[(r > 60) | ~up] = 0.0
    return pos.ffill().fillna(0.0)


STRATEGIES = {
    "sma_cross": (sma_cross, {"fast": [5, 10, 20, 30], "slow": [40, 60, 100, 200]}),
    "rsi_reversion": (rsi_reversion, {"n": [7, 14, 21], "buy": [20, 30, 35], "sell": [50, 60, 70]}),
    "donchian": (donchian, {"entry": [20, 40, 55, 100], "exit": [10, 20, 40]}),
    "trend_pullback": (trend_filter_pullback, {"trend": [100, 200], "n": [3, 5, 7], "buy": [15, 25, 35]}),
}

from .smc import SMC_SPACE, smc_sweep  # noqa: E402

STRATEGIES["smc_sweep"] = (smc_sweep, SMC_SPACE)

from .fx_strategies import FX_STRATEGIES  # noqa: E402  (long/short — лише для forex)

STRATEGIES.update(FX_STRATEGIES)
CRYPTO_NAMES = ["sma_cross", "rsi_reversion", "donchian", "trend_pullback", "smc_sweep"]
