"""Forward-контроль: чи paper/live-результати схожі на бектест. Без цього бектест — лише гіпотеза."""
import json
from pathlib import Path

STATE = Path(__file__).resolve().parent.parent / "state"
MIN_TRADES = 20


def report():
    try:
        cfg = json.loads((STATE / "strategy.json").read_text())
        st = json.loads((STATE / "fx_paper.json").read_text())
    except FileNotFoundError:
        return "Немає state/strategy.json або state/fx_paper.json — спочатку research і paper."
    log = st["log"]
    n = len(log)
    if n < MIN_TRADES:
        return f"Закритих угод {n}/{MIN_TRADES}: замало для висновків. Продовжуй paper."
    pnl = [t["pnl_pct"] for t in log]
    win = sum(x > 0 for x in pnl) / n
    eq0 = st["equity"] / (1 + sum(pnl) / 100) if pnl else st["equity"]
    peak, dd, eq = 1.0, 0.0, 1.0
    for x in pnl:
        eq *= 1 + x / 100; peak = max(peak, eq); dd = min(dd, eq / peak - 1)
    oos = cfg["oos"]
    verdict = "ТРИМАЄТЬСЯ" if sum(pnl) > 0 and dd > -0.25 else "ДЕГРАДУЄ — зупини торгівлю і перегляди стратегію"
    return (f"Угод {n}, win rate {win:.0%}, сума {sum(pnl):+.2f}%, макс. просадка {dd:.1%}\n"
            f"Бектест (OOS): return {oos['return']:+.1%}, Sharpe {oos['sharpe']:.2f}, maxDD {oos['max_dd']:.1%}\n"
            f"Вердикт: {verdict}")
