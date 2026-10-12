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
