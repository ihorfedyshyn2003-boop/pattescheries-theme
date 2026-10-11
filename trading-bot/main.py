import argparse

from bot import data, research, trader


def main():
    ap = argparse.ArgumentParser(description="Аналіз стратегій і торгівля")
    ap.add_argument("cmd", choices=["research", "paper", "live"])
    ap.add_argument("--symbol", default="BTC/USDT")
    ap.add_argument("--tf", default="1h")
    ap.add_argument("--exchange", default="binance")
    ap.add_argument("--bars", type=int, default=8000)
    ap.add_argument("--synthetic", action="store_true", help="тестові дані без мережі")
    ap.add_argument("--once", action="store_true", help="один крок (для cron)")
    a = ap.parse_args()

    if a.cmd == "research":
        df = data.synthetic(a.bars) if a.synthetic else data.fetch_ohlcv(a.symbol, a.tf, a.bars, a.exchange)
        ppy = data.periods_per_year(a.tf)
        results, bh = research.research(df, ppy)
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
        if winners:
            w = winners[0]
            params = research.pick_final_params(df, w["strategy"], ppy)
            research.save(w["strategy"], params, a.symbol, a.tf, w["oos"])
            print(f"\nЗбережено: {w['strategy']} {params} → state/strategy.json")
        else:
            print("\nРобочої стратегії не знайдено. Торгувати нічим — це чесний результат, не помилка.")
    else:
        trader.loop(a.exchange, live=a.cmd == "live", once=a.once)


if __name__ == "__main__":
    main()
