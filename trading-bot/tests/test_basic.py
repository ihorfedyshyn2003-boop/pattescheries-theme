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


def test_smc_no_lookahead_and_risk():
    from bot import smc
    df = data.synthetic(4000, start=1.08, vol=0.0006)
    s1, p1 = smc.smc_sweep(df, N=20, rr=2, trend=0, disp=0.0, session=False)
    assert (p1 != 0).any()
    df2 = df.copy(); df2.iloc[-50:, :4] *= 1.05           # змінюємо лише майбутнє
    s2, _ = smc.smc_sweep(df2, N=20, rr=2, trend=0, disp=0.0, session=False)
    assert np.allclose(s1.iloc[:-52], s2.iloc[:-52])       # минуле не змінилось
    assert s1.min() > -0.06                                 # одна свічка не з'їдає >6% (ризик 0.5%, плече ≤10)


def test_fx_paper_cycle(tmp_path, monkeypatch):
    from bot import fx
    monkeypatch.setattr(fx, "STATE", tmp_path)
    p = fx.Paper(); df = data.synthetic(600, start=1.08, vol=0.0006)
    t = p.open(1, df.close.iloc[-1] * 0.998, 2.0, df.close.iloc[-1])
    assert t and t["tp"] > t["entry"]
    p.manage(df); p.save()
    assert (tmp_path / "fx_paper.json").exists()


def test_analysis_report():
    from bot import analysis
    df = data.synthetic(5000, start=1.08, vol=0.0006)
    txt = analysis.report(df)
    assert "після sweep" in txt and "Висновок" in txt
