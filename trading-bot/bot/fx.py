"""Forex (EUR/USD) торгівля за збереженою SMC-стратегією: paper та live через OANDA REST v20.
Paper: віртуальний рахунок, стопи/тейки перевіряються по свічках. Live: OANDA_TOKEN + OANDA_ACCOUNT_ID,
стоп і тейк ставляться брокеру разом з ордером (угода захищена навіть якщо бот вимкнений)."""
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .data import TF_MS, _oanda_candles
from .smc import setups

STATE = Path(__file__).resolve().parent.parent / "state"
RISK = 0.005          # 0.5% капіталу ризику на угоду
MAX_LEV = 10.0
DAILY_LOSS_STOP = 0.03
COST = 0.00005        # спред/проковзування на сторону (частка нотіоналу) для paper


def load_cfg():
    p = STATE / "strategy.json"
    cfg = json.loads(p.read_text()) if p.exists() else None
    if not cfg or cfg.get("market") != "forex":
        raise SystemExit("Немає forex-стратегії: python main.py research --market forex ...")
    return cfg


def latest_setup(df, params):
    p = {k: v for k, v in params.items() if k != "rr"}
    d, sl = setups(df, **p)
    return int(d[-1]), float(sl[-1])


class Paper:
    def __init__(self, cash=10_000.0):
        self.path = STATE / "fx_paper.json"
        STATE.mkdir(exist_ok=True)
        self.s = json.loads(self.path.read_text()) if self.path.exists() else \
            {"equity": cash, "trade": None, "last_ts": None, "day": "", "day_start": cash, "log": []}

    def equity(self): return self.s["equity"]

    def manage(self, df):
        """Перевірка відкритої угоди по свічках після останньої перевірки."""
        t = self.s["trade"]
        last = pd.Timestamp(self.s["last_ts"]) if self.s["last_ts"] else df.index[-2]
        for ts, bar in df[df.index > last].iterrows():
            if not t:
                break
            hit_sl = bar.low <= t["sl"] if t["side"] > 0 else bar.high >= t["sl"]
            hit_tp = bar.high >= t["tp"] if t["side"] > 0 else bar.low <= t["tp"]
            if hit_sl or hit_tp:
                px = t["sl"] if hit_sl else t["tp"]
                pnl = t["side"] * t["size"] * (px / t["entry"] - 1) - COST * t["size"]
                self.s["equity"] *= 1 + pnl
                self.s["log"].append({"t": str(ts), "result": "SL" if hit_sl else "TP", "pnl_pct": pnl * 100,
                                      "equity": self.s["equity"]})
                self.s["trade"] = t = None
        self.s["last_ts"] = str(df.index[-1])

    def open(self, side, sl, rr, px):
        dist = (px - sl) * side
        if dist <= 0:
            return None
        size = min(RISK / (dist / px), MAX_LEV)
        self.s["equity"] *= 1 - COST * size
        self.s["trade"] = {"side": side, "entry": px, "sl": sl, "tp": px + side * rr * dist, "size": size}
        return self.s["trade"]

    def save(self): self.path.write_text(json.dumps(self.s, indent=2))


class Oanda:
    def __init__(self):
        import requests
        self.tok, self.acc = os.environ.get("OANDA_TOKEN"), os.environ.get("OANDA_ACCOUNT_ID")
        if not self.tok or not self.acc:
            raise SystemExit("Задай OANDA_TOKEN і OANDA_ACCOUNT_ID (спочатку demo-рахунок!)")
        self.host = "api-fxtrade.oanda.com" if os.environ.get("OANDA_LIVE") == "1" else "api-fxpractice.oanda.com"
        self.rq = requests

    def _req(self, method, path, **kw):
        r = self.rq.request(method, f"https://{self.host}/v3/accounts/{self.acc}{path}",
                            headers={"Authorization": f"Bearer {self.tok}"}, timeout=20, **kw)
        r.raise_for_status()
        return r.json()

    def nav(self): return float(self._req("GET", "/summary")["account"]["NAV"])

    def has_trade(self, instrument):
        return any(t["instrument"] == instrument for t in self._req("GET", "/openTrades")["trades"])

    def open(self, instrument, side, sl, rr, px):
        dist = (px - sl) * side
        if dist <= 0:
            return None
        nav = self.nav()
        notional = min(RISK * nav / (dist / px), MAX_LEV * nav)   # у USD
        units = int(notional / px) * side
        tp = px + side * rr * dist
        order = {"order": {"type": "MARKET", "instrument": instrument, "units": str(units),
                           "timeInForce": "FOK", "positionFill": "DEFAULT",
                           "stopLossOnFill": {"price": f"{sl:.5f}"}, "takeProfitOnFill": {"price": f"{tp:.5f}"}}}
        return self._req("POST", "/orders", json=order)


def run(live=False, once=False):
    cfg = load_cfg()
    instr, tf, params = cfg["instrument"], cfg["timeframe"], cfg["params"]
    rr = params["rr"]
    broker = Oanda() if live else Paper()
    if live and input("LIVE: торгівля через OANDA. Введи 'yes': ") != "yes":
        raise SystemExit("Скасовано")
    while True:
        try:
            df = _oanda_candles(instr, tf, 0, last_n=500)
            side, sl = latest_setup(df, params)
            px = float(df.close.iloc[-1])
            if live:
                nav, day = broker.nav(), time.strftime("%F")
                st = json.loads((STATE / "fx_live.json").read_text()) if (STATE / "fx_live.json").exists() else {}
                if st.get("day") != day:
                    st = {"day": day, "start": nav}
                    STATE.mkdir(exist_ok=True); (STATE / "fx_live.json").write_text(json.dumps(st))
                blocked = nav < st["start"] * (1 - DAILY_LOSS_STOP)
                if side and not blocked and not broker.has_trade(instr) and df.index[-1] != st.get("last_signal"):
                    print("ORDER:", broker.open(instr, side, sl, rr, px))
                    st["last_signal"] = str(df.index[-1]); (STATE / "fx_live.json").write_text(json.dumps(st))
                print(time.strftime("%F %T"), f"NAV={nav:.2f} сигнал={side} стоп-день={'ТАК' if blocked else 'ні'}")
            else:
                broker.manage(df)
                day = time.strftime("%F")
                if broker.s["day"] != day:
                    broker.s["day"], broker.s["day_start"] = day, broker.equity()
                blocked = broker.equity() < broker.s["day_start"] * (1 - DAILY_LOSS_STOP)
                opened = None
                if side and not broker.s["trade"] and not blocked and df.index[-1] != broker.s.get("sig"):
                    opened = broker.open(side, sl, rr, px); broker.s["sig"] = str(df.index[-1])
                broker.save()
                print(time.strftime("%F %T"), f"equity={broker.equity():.2f} сигнал={side} відкрито={opened}")
        except Exception as e:
            print("помилка:", e)
        if once:
            break
        step = TF_MS[tf] / 1000
        time.sleep(step - time.time() % step + 5)
