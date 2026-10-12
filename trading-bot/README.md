# trading-bot

```
pip install -r requirements.txt
python main.py research --symbol BTC/USDT --tf 1h      # дані з біржі, walk-forward, відбір
python main.py research --synthetic                    # перевірка коду без мережі
python main.py paper                                   # віртуальна торгівля (loop; --once для cron)
EXCHANGE_API_KEY=.. EXCHANGE_API_SECRET=.. python main.py live   # реальні гроші
```
Ризик: ≤25% капіталу в позиції, денний стоп -5%, long-only спот. Ключ біржі — без права виводу.
Робочою стратегія вважається лише якщо проходить 5 перевірок поза вибіркою (див. `research.verdict`).
Спершу ≥2–4 тижні `paper`, потім live з мінімальною сумою.

## Forex EUR/USD + Smart Money (SMC)
Правила SMC-сетапу — у `bot/smc.py` (sweep ліквідності + displacement + bias по EMA + сесія, SL за sweep, TP = rr×ризик).
```
# 1. Дані: OANDA demo-токен (безкоштовно) або CSV з Dukascopy/MT5
export OANDA_TOKEN=...      # practice-токен
python main.py research --market forex --symbol EUR_USD --tf 1h --bars 20000
python main.py research --market forex --csv eurusd_1h.csv
# 2. Paper (virtual) -> потім demo-рахунок OANDA -> лише тоді live
python main.py paper --market forex
export OANDA_ACCOUNT_ID=...; python main.py live --market forex     # OANDA_LIVE=1 для реального рахунку
```
Ризик: 0.5% на угоду, плече ≤10, денний стоп -3%. SL/TP ставляться брокеру разом з ордером.

## Вивчення ринку
`python main.py analyze --market forex --csv eurusd_1h.csv` → `state/market_report.md`:
волатильність за годинами, що ціна робить після sweep (з t-статистикою), інерція/відскок за режимами.
Запускай щомісяця на нових даних: якщо висновки змінились — стратегія застаріла.

## Контроль
`python main.py monitor` — порівняння paper-результатів з бектестом (після ≥20 угод: ТРИМАЄТЬСЯ / ДЕГРАДУЄ).
Прохідний бектест також вимагає bootstrap p<0.10 (відсіює везіння).

## Стратегії з відкритих джерел (forex, long/short) — `bot/fx_strategies.py`
`local_hours` (Breedon & Ranaldo), `fix_flow` (фіксинг WM/Reuters 16:00 London), `london_breakout`,
`tsmom` (Moskowitz–Ooi–Pedersen), `asian_fade`. Тест лише частини:
```
python main.py research --market forex --csv data/eurusd_1h.csv --strategies london_breakout,local_hours
```
Результат на EUR/USD 1h 2015–2025 (FXCM): жодна не пройшла walk-forward. Найближча — `london_breakout`
(азія до 07 UTC, вихід 20 UTC): Sharpe нетто 0.63 з фіксованими параметрами, але 2020–25 > 2015–20
і при подвоєних витратах ≈0. `local_hours` працював до 2020, потім зник.

### ICT / Smart Money моделі (як у SMC-школах, напр. DarkTrader) — `asia_sweep`, `fvg_retest`
Школи не публікують точних правил, тож це формалізація типових моделей курсу: зняття азійського
high/low у London killzone з поверненням у діапазон, і лімітний вхід з FVG за HTF-bias (EMA).
EUR/USD 1h 2015–2025: Sharpe брутто ≈0 (переваги немає ще до витрат), нетто −0.4…−1.3, walk-forward −45…−65%.
