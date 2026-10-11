import numpy as np
import pandas as pd


def run(df, position, ppy, fee=0.001, slippage=0.0005):
    """Векторний бектест. fee/slippage — частка від обороту на кожну зміну позиції."""
    pos = position.shift(1).fillna(0.0)               # сигнал з закритої свічки -> виконання на наступній
    ret = df.close.pct_change().fillna(0.0)
    cost = pos.diff().abs().fillna(pos.abs()) * (fee + slippage)
    strat = pos * ret - cost
    return strat, pos


def metrics(strat, pos, ppy):
    eq = (1 + strat).cumprod()
    n = len(strat)
    tot = eq.iloc[-1] - 1 if n else 0.0
    std = strat.std()
    sharpe = strat.mean() / std * np.sqrt(ppy) if std and std > 0 else 0.0
    dd = (eq / eq.cummax() - 1).min() if n else 0.0
    trades = int(((pos != pos.shift(1).fillna(0.0)) & (pos != 0)).sum())
    return {"return": float(tot), "sharpe": float(sharpe), "max_dd": float(dd),
            "trades": trades, "exposure": float(pos.mean()) if n else 0.0}


def buy_hold(df, ppy):
    r = df.close.pct_change().fillna(0.0)
    return metrics(r, pd.Series(1.0, index=df.index), ppy)
