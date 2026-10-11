import numpy as np
from bot import backtest, data, research, strategies


def test_no_lookahead():
    df = data.synthetic(1500)
    pos = strategies.sma_cross(df, 10, 40)
    cut = pos.copy(); cut.iloc[-1] = 1 - cut.iloc[-1]   # зміна останнього сигналу
    s1, _ = backtest.run(df, pos, 8760); s2, _ = backtest.run(df, cut, 8760)
    assert np.allclose(s1.iloc[:-1], s2.iloc[:-1]) and np.allclose(s1.iloc[-1], s2.iloc[-1])


def test_costs_reduce_return():
    df = data.synthetic(2000)
    pos = strategies.rsi_reversion(df)
    free, _ = backtest.run(df, pos, 8760, fee=0, slippage=0)
    paid, _ = backtest.run(df, pos, 8760)
    assert (1 + paid).prod() <= (1 + free).prod()


def test_walk_forward_runs():
    df = data.synthetic(5000)
    res, bh = research.research(df, 8760)
    assert res and all("working" in r for r in res)
