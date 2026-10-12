"""Forex-стратегії з відкритих джерел (наукові статті + популярні інтрадей-сетапи), long/short.

Позиція на свічці t (відома на її закритті) працює на свічці t+1 — це робить backtest.run().
Час у даних — UTC, індекс = час відкриття свічки.

1. local_hours   — Breedon & Ranaldo (2013), "Intraday patterns in FX returns and order flow":
                   валюта дешевшає у свої робочі години → шорт EUR/USD у європейські години,
                   лонг у американські.
2. fix_flow      — Krohn, Mueller & Whelan, "FX fixings and returns around the clock" / Evans (2018):
                   долар дорожчає перед лондонським фіксингом WM/Reuters 16:00 (London time)
                   і відкочується після нього.
3. london_breakout — популярний ретейл-сетап: пробій азійського діапазону на відкритті Лондона,
                   стоп за протилежним краєм, вихід у кінці дня.
4. tsmom         — Moskowitz, Ooi & Pedersen (2012), "Time series momentum": знак дохідності
                   за минулий період, перерахунок раз на добу.
5. asian_fade    — популярний «нічний скальпер»: у тиху азійську сесію торгувати повернення до середнього.
"""
import numpy as np
import pandas as pd

from .smc import atr


def _next_hour(df, tz=None):
    """Година НАСТУПНОЇ свічки (на ній діятиме позиція, виставлена зараз)."""
    idx = df.index + pd.Timedelta(hours=1)
    if tz:
        idx = idx.tz_localize("UTC").tz_convert(tz)
    return np.asarray(idx.hour)


def _in(h, start, end):
    return (h >= start) & (h < end) if start < end else (h >= start) | (h < end)


def local_hours(df, eu_start=7, eu_end=12, us_start=13, us_end=20, side="both"):
    h = _next_hour(df)
    pos = np.zeros(len(df))
    if side in ("both", "eu"):
        pos[_in(h, eu_start, eu_end)] = -1.0     # EUR дешевшає у європейські години
    if side in ("both", "us"):
        pos[_in(h, us_start, us_end)] = 1.0      # USD дешевшає в американські години
    return pd.Series(pos, index=df.index)


def fix_flow(df, before=2, after=2, mode="both"):
    """Фіксинг о 16:00 за Лондоном (з урахуванням літнього часу)."""
    h = _next_hour(df, "Europe/London")
    pos = np.zeros(len(df))
    if mode in ("both", "before"):
        pos[_in(h, 16 - before, 16)] = -1.0      # перед фіксингом купують долар
    if mode in ("both", "after"):
        pos[_in(h, 16, 16 + after)] = 1.0        # після — відкат
    return pd.Series(pos, index=df.index)


def london_breakout(df, cost=0.00005, risk=0.005, max_lev=10.0, asia_end=7, exit_hour=16, rr=0.0):
    """Діапазон 00:00–asia_end UTC; після нього перший пробій high/low → вхід за ціною пробою,
    стоп — протилежний край діапазону, тейк rr×ризик (0 = без тейку), вихід о exit_hour."""
    o, h, l, c = (df[k].values for k in ("open", "high", "low", "close"))
    hour = np.asarray(df.index.hour)
    day = np.asarray(df.index.normalize().asi8)
    n = len(df)
    strat, pos = np.zeros(n), np.zeros(n)
    cur_day, hi, lo, traded = None, -np.inf, np.inf, False
    in_trade, side, sl, tp, size, prev = False, 0, 0.0, 0.0, 0.0, 0.0
    for t in range(n):
        if day[t] != cur_day:
            cur_day, hi, lo, traded = day[t], -np.inf, np.inf, False
            if in_trade:                      # дані з розривом — закриваємо по open
                strat[t] += side * size * (o[t] / prev - 1) - cost * size
                in_trade = False
        if hour[t] < asia_end:
            hi, lo = max(hi, h[t]), min(lo, l[t])
            continue
        if in_trade:
            pos[t] = side
            hit_sl = (l[t] <= sl) if side > 0 else (h[t] >= sl)
            hit_tp = rr > 0 and ((h[t] >= tp) if side > 0 else (l[t] <= tp))
            if hit_sl or hit_tp:
                px = sl if hit_sl else tp
                strat[t] += side * size * (px / prev - 1) - cost * size
                in_trade = False
            elif hour[t] >= exit_hour - 1:   # закриття на close останньої свічки
                strat[t] += side * size * (c[t] / prev - 1) - cost * size
                in_trade = False
            else:
                strat[t] += side * size * (c[t] / prev - 1)
                prev = c[t]
            continue
        if traded or hour[t] >= exit_hour - 1 or not np.isfinite(hi) or hi <= lo:
            continue
        up, dn = h[t] > hi, l[t] < lo
        if up == dn:                          # нема пробою або пробій в обидва боки — пропуск
            if up:
                traded = True
            continue
        side = 1 if up else -1
        entry = max(o[t], hi) if up else min(o[t], lo)
        stop = lo if up else hi
        dist = abs(entry - stop)
        size = min(risk / (dist / entry), max_lev)
        sl, tp = stop, entry + side * rr * dist
        traded, in_trade, pos[t] = True, True, side
        strat[t] -= cost * size
        if (l[t] <= sl) if side > 0 else (h[t] >= sl):       # стоп у тій самій свічці (песимістично)
            strat[t] += side * size * (sl / entry - 1) - cost * size
            in_trade = False
        else:
            strat[t] += side * size * (c[t] / entry - 1)
            prev = c[t]
    return pd.Series(strat, index=df.index), pd.Series(pos, index=df.index)


london_breakout.simulated = True


def tsmom(df, lookback_days=60, long_only=False):
    """Знак дохідності за lookback_days; сигнал оновлюється раз на добу (о 21:00 UTC)."""
    lb = lookback_days * 24
    sig = np.sign(df.close / df.close.shift(lb) - 1)
    if long_only:
        sig = sig.clip(lower=0)
    daily = pd.Series(np.where(df.index.hour == 21, sig, np.nan), index=df.index)
    return daily.ffill().fillna(0.0)


tsmom.warmup = 260 * 24


def asian_fade(df, n=20, z_in=2.0, start=22, end=5):
    """Лише в азійську сесію: ціна відійшла від середньої на z_in·σ → ставимо на повернення;
    вихід — на середній або в кінці сесії."""
    m = df.close.rolling(n).mean()
    s = df.close.rolling(n).std()
    z = ((df.close - m) / s).values
    ok = _in(_next_hour(df), start, end)
    pos = np.zeros(len(df))
    cur = 0.0
    for t in range(len(df)):
        if not ok[t] or not np.isfinite(z[t]):
            cur = 0.0
        elif cur == 0.0:
            cur = -1.0 if z[t] > z_in else (1.0 if z[t] < -z_in else 0.0)
        elif (cur > 0 and z[t] >= 0) or (cur < 0 and z[t] <= 0):
            cur = 0.0
        pos[t] = cur
    return pd.Series(pos, index=df.index)


FX_STRATEGIES = {
    "local_hours": (local_hours, {"eu_start": [6, 7, 8], "eu_end": [10, 12], "us_start": [13, 16],
                                  "us_end": [18, 20], "side": ["both", "eu", "us"]}),
    "fix_flow": (fix_flow, {"before": [1, 2, 3], "after": [1, 2, 3], "mode": ["both", "before", "after"]}),
    "london_breakout": (london_breakout, {"asia_end": [6, 7, 8], "exit_hour": [12, 16, 20], "rr": [0.0, 1.0, 2.0]}),
    "tsmom": (tsmom, {"lookback_days": [20, 60, 120, 250], "long_only": [False]}),
    "asian_fade": (asian_fade, {"n": [10, 20, 40], "z_in": [1.5, 2.0, 2.5], "start": [21, 22], "end": [4, 6]}),
}


# ---------- ICT / Smart Money моделі (як їх викладають SMC-школи, напр. DarkTrader) ----------
# Публічних точних правил школа не дає — це МОЯ формалізація типових моделей курсу:
# «Asia range → London sweep (Judas swing)» і «вхід з FVG (fair value gap) за HTF-bias».

def _close_trade(strat, t, side, size, px, prev, cost):
    strat[t] += side * size * (px / prev - 1) - cost * size


def asia_sweep(df, cost=0.00005, risk=0.005, max_lev=10.0, asia_end=6, kill_end=11, target="opposite",
               rr=2.0, exit_hour=20, buf=0.1):
    """Азійський діапазон 00:00–asia_end UTC. У London killzone (asia_end–kill_end) ціна знімає high (low)
    Азії і свічка закривається НАЗАД у діапазон → шорт (лонг) з open наступної свічки.
    SL: за екстремум sweep + buf·ATR. TP: протилежний край Азії (target='opposite') або rr×ризик.
    Одна угода на день, примусовий вихід о exit_hour."""
    a = atr(df).values
    o, h, l, c = (df[k].values for k in ("open", "high", "low", "close"))
    hour = np.asarray(df.index.hour)
    day = np.asarray(df.index.normalize().asi8)
    n = len(df)
    strat, pos = np.zeros(n), np.zeros(n)
    cur, hi, lo, done = None, -np.inf, np.inf, False
    pend, in_trade, side, sl, tp, size, prev = 0, False, 0, 0.0, 0.0, 0.0, 0.0
    pend_sl = 0.0
    for t in range(n):
        if day[t] != cur:
            if in_trade:
                _close_trade(strat, t, side, size, o[t], prev, cost); in_trade = False
            cur, hi, lo, done, pend = day[t], -np.inf, np.inf, False, 0
        if pend and not in_trade:                       # вхід з open цієї свічки
            side, entry = pend, o[t]
            dist = (pend_sl - entry) * -side
            pend = 0
            tgt = (hi if side > 0 else lo) if target == "opposite" else entry + side * rr * dist
            if dist > 0 and (tgt - entry) * side > 0:
                size = min(risk / (dist / entry), max_lev)
                sl, tp, prev, in_trade = pend_sl, tgt, entry, True
                strat[t] -= cost * size
        if hour[t] < asia_end:
            hi, lo = max(hi, h[t]), min(lo, l[t])
            continue
        if in_trade:
            pos[t] = side
            hit_sl = (l[t] <= sl) if side > 0 else (h[t] >= sl)
            hit_tp = (h[t] >= tp) if side > 0 else (l[t] <= tp)
            if hit_sl or hit_tp:
                _close_trade(strat, t, side, size, sl if hit_sl else tp, prev, cost); in_trade = False
            elif hour[t] >= exit_hour - 1:
                _close_trade(strat, t, side, size, c[t], prev, cost); in_trade = False
            else:
                strat[t] += side * size * (c[t] / prev - 1); prev = c[t]
            continue
        if done or hour[t] >= kill_end or not np.isfinite(hi) or not np.isfinite(a[t]):
            continue
        if h[t] > hi and c[t] < hi and not (l[t] < lo):
            pend, pend_sl, done = -1, h[t] + buf * a[t], True
        elif l[t] < lo and c[t] > lo and not (h[t] > hi):
            pend, pend_sl, done = 1, l[t] - buf * a[t], True
    return pd.Series(strat, index=df.index), pd.Series(pos, index=df.index)


asia_sweep.simulated = True


def fvg_retest(df, cost=0.00005, risk=0.005, max_lev=10.0, trend=100, rr=2.0, wait=6, session=True):
    """FVG: бичачий — low[t] > high[t-2] (гап між свічками t-2 і t), ведмежий — дзеркально.
    Лише за bias (ціна вище/нижче EMA trend) і в Лондон/НЙ 07–16 UTC. Лімітний вхід на краю гепу
    (high[t-2] для лонга), якщо ціна повернеться протягом wait свічок. SL: за протилежний край
    свічки t-2 (low[t-2] для лонга). TP: rr×ризик. Обидва в одній свічці → SL."""
    o, h, l, c = (df[k].values for k in ("open", "high", "low", "close"))
    ema = df.close.ewm(span=trend, adjust=False).mean().values if trend else None
    hour = np.asarray(df.index.hour)
    n = len(df)
    strat, pos = np.zeros(n), np.zeros(n)
    order = None                                         # (side, limit, stop, expiry)
    in_trade, side, sl, tp, size, prev = False, 0, 0.0, 0.0, 0.0, 0.0
    for t in range(2, n):
        if in_trade:
            pos[t] = side
            hit_sl = (l[t] <= sl) if side > 0 else (h[t] >= sl)
            hit_tp = (h[t] >= tp) if side > 0 else (l[t] <= tp)
            if hit_sl or hit_tp:
                _close_trade(strat, t, side, size, sl if hit_sl else tp, prev, cost); in_trade = False
            else:
                strat[t] += side * size * (c[t] / prev - 1); prev = c[t]
            continue
        if order:
            s_, lim, stp, exp = order
            if t > exp:
                order = None
            elif (l[t] <= lim) if s_ > 0 else (h[t] >= lim):
                entry = min(o[t], lim) if s_ > 0 else max(o[t], lim)
                dist = (entry - stp) * s_
                order = None
                if dist > 0:
                    side, size = s_, min(risk / (dist / entry), max_lev)
                    sl, tp, in_trade, pos[t] = stp, entry + s_ * rr * dist, True, s_
                    strat[t] -= cost * size
                    if (l[t] <= sl) if side > 0 else (h[t] >= sl):     # песимістично: стоп у свічці входу
                        _close_trade(strat, t, side, size, sl, entry, cost); in_trade = False
                    else:
                        strat[t] += side * size * (c[t] / entry - 1); prev = c[t]
                continue
        ok = (not session) or (7 <= hour[t] < 16)
        up_bias = ema is None or c[t] > ema[t]
        dn_bias = ema is None or c[t] < ema[t]
        if ok and l[t] > h[t - 2] and up_bias:
            order = (1, h[t - 2], l[t - 2], t + wait)
        elif ok and h[t] < l[t - 2] and dn_bias:
            order = (-1, l[t - 2], h[t - 2], t + wait)
    return pd.Series(strat, index=df.index), pd.Series(pos, index=df.index)


fvg_retest.simulated = True

FX_STRATEGIES["asia_sweep"] = (asia_sweep, {"asia_end": [6, 7], "kill_end": [10, 12], "target": ["opposite", "rr"],
                                            "rr": [1.5, 2.0, 3.0]})
FX_STRATEGIES["fvg_retest"] = (fvg_retest, {"trend": [0, 100, 200], "rr": [1.5, 2.0, 3.0], "wait": [3, 6]})
