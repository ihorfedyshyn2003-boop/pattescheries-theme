import argparse

from bot import data, research, trader

COSTS = {"crypto": dict(fee=0.001, slippage=0.0005), "forex": dict(fee=0.00003, slippage=0.00002)}


def main():
    ap = argparse.ArgumentParser(description="Аналіз стратегій і торгівля")
    ap.add_argument("cmd", choices=["research", "analyze", "paper", "live"])
    ap.add_argument("--market", choices=["crypto", "forex"], default="crypto")
    ap.add_argument("--symbol", default=None, help="BTC/USDT або EUR_USD")
    ap.add_argument("--tf", default="1h")
    ap.add_argument("--exchange", default="binance")
    ap.add_argument("--source", default="oanda", help="forex: oanda | yahoo")
    ap.add_argument("--csv", help="forex: CSV з Dukascopy/MT5/TradingView")
    ap.add_argument("--bars", type=int, default=8000)
    ap.add_argument("--synthetic", action="store_true", help="тестові дані без мережі")
    ap.add_argument("--once", action="store_true", help="один крок (для cron)")
    a = ap.parse_args()
    fx = a.market == "forex"
    a.symbol = a.symbol or ("EUR_USD" if fx else "BTC/USDT")

    if a.cmd in ("research", "analyze"):
        if a.synthetic:
            df = data.synthetic(a.bars, start=1.08, vol=0.0006) if fx else data.synthetic(a.bars)
        elif fx:
            df = data.fetch_forex(a.symbol, a.tf, a.bars, a.source, a.csv)
        else:
            df = data.fetch_ohlcv(a.symbol, a.tf, a.bars, a.exchange)
        if a.cmd == "analyze":
            from pathlib import Path

            from bot import analysis
            txt = analysis.report(df)
            Path("state").mkdir(exist_ok=True)
            Path("state/market_report.md").write_text(txt)
            return print(txt)
        ppy = data.periods_per_year(a.tf, a.market)
        results, bh = research.research(df, ppy, **COSTS[a.market])
        print(f"Дані: {len(df)} свічок {df.index[0]} → {df.index[-1]}")
        print(f"Buy&Hold: return {bh['return']:+.1%}, Sharpe {bh['sharpe']:.2f}, maxDD {bh['max_dd']:.1%}\n")
        for r in results:
            o = r["oos"]
            print(f"{'✅' if r['working'] else '❌'} {r['strategy']:15} OOS: return {o['return']:+.1%} "
                  f"Sharpe {o['sharpe']:.2f} maxDD {o['max_dd']:.1%} угод {o['trades']} фолдів+ {o['folds_profitable']:.0%}")
            for k, v in r["checks"].items():
                if not v:
                    print(f"     ✗ {k}")
        winners = [r for r in results if r["working"]]
        if fx:   # у forex торгуємо лише SMC-реалізацією (є шорт і стопи)
            winners = [r for r in winners if r["strategy"] == "smc_sweep"]
        if winners:
            w = winners[0]
            params = research.pick_final_params(df, w["strategy"], ppy, **COSTS[a.market])
            research.save(w["strategy"], params, a.symbol, a.tf, w["oos"], a.market)
            print(f"\nЗбережено: {w['strategy']} {params} → state/strategy.json")
        else:
            print("\nРобочої стратегії не знайдено. Торгувати нічим — це чесний результат, не помилка.")
    elif fx:
        from bot import fx as fxbot
        fxbot.run(live=a.cmd == "live", once=a.once)
    else:
        trader.loop(a.exchange, live=a.cmd == "live", once=a.once)


if __name__ == "__main__":
    main()
