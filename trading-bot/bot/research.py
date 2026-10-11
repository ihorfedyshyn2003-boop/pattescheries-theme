"""Пошук робочої стратегії: grid search на train + чесна перевірка на невидимому test (walk-forward).
Стратегія вважається 'робочою' лише якщо тримається поза вибіркою (OOS), а не лише на історії."""
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .backtest import buy_hold, metrics, run
from .strategies import STRATEGIES

OUT = Path(__file__).resolve().parent.parent / "state"


def grid(space):
    keys = list(space)
    for vals in itertools.product(*(space[k] for k in keys)):
        p = dict(zip(keys, vals))
        if "fast" in p and p["fast"] >= p["slow"]:
            continue
        yield p


def evaluate(df, fn, params, ppy, **cost):
    strat, pos = run(df, fn(df, **params), ppy, **cost)
    return metrics(strat, pos, ppy), strat, pos


def walk_forward(df, name, ppy, train=2000, test=500, min_trades=5, **cost):
    fn, space = STRATEGIES[name]
    folds, oos = [], []
    start = 0
    while start + train + test <= len(df):
        tr = df.iloc[start:start + train]
        # warmup: тест рахуємо з хвостом train, щоб індикатори були прогріті, метрики — лише по test
        te_full = df.iloc[start + train - 300:start + train + test]
        best, best_s = None, -9
        for p in grid(space):
            m, _, _ = evaluate(tr, fn, p, ppy, **cost)
            if m["trades"] >= min_trades and m["sharpe"] > best_s:
                best, best_s = p, m["sharpe"]
        if best:
            strat, pos = run(te_full, fn(te_full, **best), ppy, **cost)
            strat, pos = strat.iloc[300:], pos.iloc[300:]
            m = metrics(strat, pos, ppy)
            folds.append({"params": best, "train_sharpe": best_s, **{f"test_{k}": v for k, v in m.items()}})
            oos.append((strat, pos))
        start += test
    if not oos:
        return None
    s = pd.concat([o[0] for o in oos]); p = pd.concat([o[1] for o in oos])
    agg = metrics(s, p, ppy)
    agg["folds"] = len(folds)
    agg["folds_profitable"] = sum(f["test_return"] > 0 for f in folds) / len(folds)
    agg["trades"] = sum(f["test_trades"] for f in folds)
    return {"strategy": name, "oos": agg, "folds": folds}


def verdict(res, bh):
    o = res["oos"]
    checks = {
        "OOS Sharpe > 0.5": o["sharpe"] > 0.5,
        "OOS прибуток > 0": o["return"] > 0,
        "≥60% фолдів у плюсі": o["folds_profitable"] >= 0.6,
        "≥20 угод (статистика)": o["trades"] >= 20,
        "просадка менша за buy&hold": o["max_dd"] > bh["max_dd"],
    }
    return all(checks.values()), checks


def research(df, ppy, **cost):
    bh = buy_hold(df, ppy)
    results = []
    for name in STRATEGIES:
        r = walk_forward(df, name, ppy, **cost)
        if r:
            ok, checks = verdict(r, bh)
            r["working"], r["checks"] = ok, checks
            results.append(r)
    results.sort(key=lambda r: r["oos"]["sharpe"], reverse=True)
    return results, bh


def pick_final_params(df, name, ppy, **cost):
    """Параметри для торгівлі: оптимум на останніх 2000 свічках."""
    fn, space = STRATEGIES[name]
    tail = df.iloc[-2000:]
    best, best_s = None, -9
    for p in grid(space):
        m, _, _ = evaluate(tail, fn, p, ppy, **cost)
        if m["trades"] >= 5 and m["sharpe"] > best_s:
            best, best_s = p, m["sharpe"]
    return best


def save(name, params, symbol, timeframe, oos):
    OUT.mkdir(exist_ok=True)
    (OUT / "strategy.json").write_text(json.dumps(
        {"strategy": name, "params": params, "symbol": symbol, "timeframe": timeframe, "oos": oos}, indent=2))
