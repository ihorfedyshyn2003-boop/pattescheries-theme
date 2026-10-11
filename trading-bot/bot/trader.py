"""Торгівля за збереженою стратегією. За замовчуванням — paper (віртуальний рахунок).
Live: лише з --live, ключами в env (EXCHANGE_API_KEY/SECRET) і ліміти ризику нижче."""
import json
import os
import time
from pathlib import Path

from .data import TF_MS, fetch_recent
from .strategies import STRATEGIES

STATE = Path(__file__).resolve().parent.parent / "state"

MAX_ALLOC = 0.25        # максимум 25% капіталу в позиції
DAILY_LOSS_STOP = 0.05  # стоп торгівлі на день при -5% від старту дня
FEE = 0.001


def load_strategy():
    p = STATE / "strategy.json"
    if not p.exists():
        raise SystemExit("Немає state/strategy.json — спочатку: python main.py research")
    return json.loads(p.read_text())


def target_position(cfg, exchange):
    df = fetch_recent(cfg["symbol"], cfg["timeframe"], exchange, n=500)
    df = df.iloc[:-1]  # остання свічка ще не закрита
    fn, _ = STRATEGIES[cfg["strategy"]]
    return float(fn(df, **cfg["params"]).iloc[-1]), float(df.close.iloc[-1])


class Paper:
    def __init__(self, cash=1000.0):
        self.path = STATE / "paper.json"
        STATE.mkdir(exist_ok=True)
        self.s = json.loads(self.path.read_text()) if self.path.exists() else \
            {"cash": cash, "qty": 0.0, "start_equity": cash, "day": time.strftime("%Y-%m-%d"), "log": []}

    def equity(self, px): return self.s["cash"] + self.s["qty"] * px

    def rebalance(self, want_long, px):
        eq = self.equity(px)
        if want_long and self.s["qty"] == 0:
            spend = eq * MAX_ALLOC
            self.s["qty"] = spend * (1 - FEE) / px; self.s["cash"] -= spend; side = "BUY"
        elif not want_long and self.s["qty"] > 0:
            self.s["cash"] += self.s["qty"] * px * (1 - FEE); self.s["qty"] = 0.0; side = "SELL"
        else:
            return None
        self.s["log"].append({"t": time.strftime("%F %T"), "side": side, "px": px, "equity": self.equity(px)})
        return side

    def save(self): self.path.write_text(json.dumps(self.s, indent=2))


class Live:
    def __init__(self, exchange, symbol):
        import ccxt
        key, sec = os.environ.get("EXCHANGE_API_KEY"), os.environ.get("EXCHANGE_API_SECRET")
        if not key or not sec:
            raise SystemExit("Задай EXCHANGE_API_KEY / EXCHANGE_API_SECRET (ключ БЕЗ права виводу коштів!)")
        self.ex = getattr(ccxt, exchange)({"apiKey": key, "secret": sec, "enableRateLimit": True})
        self.symbol, self.base, self.quote = symbol, *symbol.split("/")

    def equity(self, px):
        b = self.ex.fetch_balance()
        return b[self.quote]["total"] + b[self.base]["total"] * px

    def holding(self): return self.ex.fetch_balance()[self.base]["total"] > 0

    def rebalance(self, want_long, px):
        have = self.holding()
        if want_long and not have:
            amt = float(self.ex.amount_to_precision(self.symbol, self.equity(px) * MAX_ALLOC / px))
            self.ex.create_market_buy_order(self.symbol, amt); return "BUY"
        if not want_long and have:
            amt = float(self.ex.amount_to_precision(self.symbol, self.ex.fetch_balance()[self.base]["free"]))
            self.ex.create_market_sell_order(self.symbol, amt); return "SELL"


def step(broker, cfg, exchange, paper):
    want, px = target_position(cfg, exchange)
    eq = broker.equity(px)
    if paper:
        today = time.strftime("%Y-%m-%d")
        if broker.s["day"] != today:
            broker.s["day"], broker.s["start_equity"] = today, eq
        if eq < broker.s["start_equity"] * (1 - DAILY_LOSS_STOP):
            want = 0.0
            print("ДЕННИЙ СТОП-ЛОС: тільки закриваємо позицію")
    side = broker.rebalance(bool(want), px)
    print(f"{time.strftime('%F %T')} px={px:.2f} сигнал={'LONG' if want else 'FLAT'} equity={eq:.2f} дія={side or '-'}")
    if paper:
        broker.save()


def loop(exchange="binance", live=False, once=False):
    cfg = load_strategy()
    broker = Live(exchange, cfg["symbol"]) if live else Paper()
    if live and input("LIVE: реальні гроші. Введи 'yes' для підтвердження: ") != "yes":
        raise SystemExit("Скасовано")
    while True:
        try:
            step(broker, cfg, exchange, paper=not live)
        except Exception as e:  # мережа/біржа не мають валити бота
            print("помилка:", e)
        if once:
            break
        time.sleep(TF_MS[cfg["timeframe"]] / 1000)
