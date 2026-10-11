"""Статистичне вивчення ринку: що ціна робить у різні години, після sweep, у різних режимах.
Кожен висновок має t-статистику; |t| < 3 вважаємо шумом (багато гіпотез -> багато випадкових 'закономірностей')."""
import numpy as np
import pandas as pd

from .smc import atr, setups

T_MIN = 3.0


def tstat(x):
    x = pd.Series(x).dropna()
    return float(x.mean() / (x.std() / np.sqrt(len(x)))) if len(x) > 2 and x.std() > 0 else 0.0


def by_hour(df):
    r = df.close.pct_change() * 1e4          # б.п.
    g = r.groupby(df.index.hour)
    return pd.DataFrame({"vol_bp": g.apply(lambda x: x.abs().mean()), "mean_bp": g.mean(),
                         "t": g.apply(tstat), "n": g.size()})


def by_weekday(df):
    r = df.close.pct_change() * 1e4
    g = r.groupby(df.index.dayofweek)
    return pd.DataFrame({"vol_bp": g.apply(lambda x: x.abs().mean()), "mean_bp": g.mean(), "t": g.apply(tstat)})


def after_sweep(df, horizons=(1, 4, 12, 24), N=20):
    """Дохідність після sweep у напрямку 'розвороту' (long після sweep низів, short після sweep верхів), б.п."""
    d, _ = setups(df, N=N, trend=0, disp=0.0, session=False)
    rows = []
    for h in horizons:
        fwd = (df.close.shift(-h) / df.close - 1) * 1e4 * pd.Series(d, index=df.index)
        ev = fwd[d != 0].dropna()
        base = (df.close.shift(-h) / df.close - 1).abs().mean() * 1e4
        rows.append({"horizon": h, "n": len(ev), "mean_bp": ev.mean(), "hit_rate": (ev > 0).mean(),
                     "t": tstat(ev), "typical_move_bp": base})
    return pd.DataFrame(rows).set_index("horizon")


def regimes(df, win=50):
    """Режим за efficiency ratio: ~1 = чистий тренд, ~0 = флет. Автокореляція дохідностей = momentum(+)/відскок(−)."""
    r = df.close.pct_change()
    er = (df.close - df.close.shift(win)).abs() / df.close.diff().abs().rolling(win).sum()
    q = pd.qcut(er, 3, labels=["флет", "змішаний", "тренд"])
    nxt = r.shift(-1)
    out = []
    for name in ["флет", "змішаний", "тренд"]:
        m = (q == name)
        # знак попередньої години * наступна дохідність: >0 => інерція, <0 => відскок
        prod = (np.sign(r) * nxt)[m].dropna() * 1e4
        out.append({"режим": name, "n": len(prod), "інерція_bp": prod.mean(), "t": tstat(prod)})
    return pd.DataFrame(out).set_index("режим")


def vol_clustering(df):
    a = (df.close.pct_change().abs()).dropna()
    return float(a.autocorr(1)), float(a.autocorr(24))


def report(df):
    L = ["# Звіт про поведінку ринку", f"Свічок: {len(df)}  {df.index[0]} → {df.index[-1]}", ""]
    h = by_hour(df)
    L += ["## Волатильність за годинами (UTC), б.п. за свічку",
          "Найактивніші: " + ", ".join(f"{i}:00 ({v:.1f})" for i, v in h.vol_bp.nlargest(4).items()),
          "Найтихіші: " + ", ".join(f"{i}:00 ({v:.1f})" for i, v in h.vol_bp.nsmallest(4).items()), ""]
    sig = h[h.t.abs() >= T_MIN]
    L += ["Години зі значущим напрямком (|t|≥3): " + (", ".join(f"{i}:00 {r.mean_bp:+.2f}bp t={r.t:.1f}"
                                                              for i, r in sig.iterrows()) or "немає — напрямок за годинами = шум"), ""]
    L += ["## Що ціна робить після sweep (розворот у бік сетапу)", after_sweep(df).round(3).to_string(), ""]
    L += ["## Режими: інерція (+) чи відскок (−) наступної свічки", regimes(df).round(3).to_string(), ""]
    a1, a24 = vol_clustering(df)
    L += [f"## Кластеризація волатильності: автокореляція |ret| lag1={a1:.2f}, lag24={a24:.2f}"
          " (>0.1 = після бурхливої свічки чекай ще)", ""]
    s = after_sweep(df)
    pro, anti = s[s.t >= T_MIN], s[s.t <= -T_MIN]
    L += ["## Висновок"]
    if len(pro):
        L.append("ЗА стратегію: після sweep ціна значуще розвертається на горизонтах " + ", ".join(map(str, pro.index)))
    if len(anti):
        L.append("ПРОТИ стратегії: після sweep ціна значуще продовжує рух ПРОТИ розвороту на горизонтах "
                 + ", ".join(map(str, anti.index)) + " (ринок продовжує пробій — можливо, інвертувати/не торгувати)")
    if not len(pro) and not len(anti):
        L.append("Після sweep НЕМАЄ значущого ефекту — ідея SMC не підтверджена на цих даних.")
    return "\n".join(L)
