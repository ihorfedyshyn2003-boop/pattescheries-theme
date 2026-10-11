"""Smart Money Concepts (SMC), формалізація для EUR/USD. Це МОЯ строга версія ідеї, не авторська.

Сетап (long; short дзеркально):
  1. Зняття ліквідності (sweep): low свічки пробиває мінімум останніх N свічок (стопи покупців вибито)...
  2. ...але закриття повертається вище цього рівня (відхилення, "ложний пробій").
  3. Displacement: тіло свічки бичаче й > disp * ATR (імпульс інституціоналів).
  4. Фільтр тренду (HTF-bias): ціна вище/нижче EMA(trend) — торгуємо лише за bias; trend=0 вимикає.
  5. Сесія: лише Лондон/Нью-Йорк (07–16 UTC), коли є ліквідність.
Вхід: open наступної свічки. Стоп: за екстремум sweep ± 0.1 ATR. Тейк: rr × ризик.
Ризик на угоду — фіксована частка капіталу (risk), плече обмежене max_lev.
Якщо SL і TP в одній свічці — рахуємо SL (песимістично)."""
import numpy as np
import pandas as pd


def atr(df, n=14):
    pc = df.close.shift(1)
    tr = pd.concat([df.high - df.low, (df.high - pc).abs(), (df.low - pc).abs()], axis=1).max(axis=1)
    return tr.rolling(n).mean()


def setups(df, N=20, trend=100, disp=0.5, session=True):
    """Повертає (dir, sl): dir[t] ∈ {-1,0,1} відомий на закритті свічки t; sl — рівень стопу."""
    a = atr(df)
    ref_lo = df.low.rolling(N).min().shift(1)
    ref_hi = df.high.rolling(N).max().shift(1)
    body = df.close - df.open
    long_ = (df.low < ref_lo) & (df.close > ref_lo) & (body > disp * a)
    short_ = (df.high > ref_hi) & (df.close < ref_hi) & (-body > disp * a)
    if trend:
        ema = df.close.ewm(span=trend, adjust=False).mean()
        long_ &= df.close > ema
        short_ &= df.close < ema
    if session:
        ok = (df.index.hour >= 7) & (df.index.hour < 16)
        long_ &= ok
        short_ &= ok
    d = pd.Series(0, index=df.index)
    d[long_] = 1
    d[short_] = -1
    sl = pd.Series(np.nan, index=df.index)
    sl[long_] = (df.low - 0.1 * a)[long_]
    sl[short_] = (df.high + 0.1 * a)[short_]
    return d.values, sl.values


def smc_sweep(df, cost=0.00005, risk=0.005, max_lev=10.0, rr=2.0, **kw):
    """Симуляція угод. Повертає (дохідність на капітал по свічках, позиція). cost — частка нотіоналу за сторону."""
    d, sl_arr = setups(df, **kw)
    o, h, l, c = (df[k].values for k in ("open", "high", "low", "close"))
    n = len(df)
    strat, pos = np.zeros(n), np.zeros(n)
    in_trade, side, sl, tp, size, prev = False, 0, 0.0, 0.0, 0.0, 0.0
    for t in range(1, n):
        if not in_trade and d[t - 1] != 0:
            side, entry, stop = int(d[t - 1]), o[t], sl_arr[t - 1]
            dist = (entry - stop) * side
            if dist > 0:   # якщо відкриття вже за стопом — пропускаємо
                size = min(risk / (dist / entry), max_lev)
                sl, tp, prev, in_trade = stop, entry + side * rr * dist, entry, True
                strat[t] -= cost * size
        if in_trade:
            hit_sl = (l[t] <= sl) if side > 0 else (h[t] >= sl)
            hit_tp = (h[t] >= tp) if side > 0 else (l[t] <= tp)
            pos[t] = side
            if hit_sl or hit_tp:
                px = sl if hit_sl else tp
                strat[t] += side * size * (px / prev - 1) - cost * size
                in_trade = False
            else:
                strat[t] += side * size * (c[t] / prev - 1)
                prev = c[t]
    return pd.Series(strat, index=df.index), pd.Series(pos, index=df.index)


smc_sweep.simulated = True
SMC_SPACE = {"N": [10, 20, 40], "rr": [1.5, 2.0, 3.0], "trend": [0, 100, 200],
             "disp": [0.0, 0.5], "session": [False, True]}
