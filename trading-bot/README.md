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
