import numpy as np
import pandas as pd
import pytest

from pmes.analysis import (adjustment_profile, cross_correlation, detect_shocks, event_paths,
                           fit_sensitivity, lead_lag_regressions)
from pmes.config import Config
from pmes.data import align
from pmes.strategy import trade_events
from pmes.synthetic import simulate_pair


@pytest.fixture(scope="module")
def lagged():
    prob, asset = simulate_pair(beta=15.0, p0=0.45, n=8000, seed=3)
    return align(prob, asset)


@pytest.fixture(scope="module")
def no_lag():
    # Same market but the probability responds fully on the shock bar.
    prob, asset = simulate_pair(beta=15.0, p0=0.45, n=8000, seed=3, lag_weights=np.array([1.0]))
    return align(prob, asset)


def test_align_uses_last_probability_at_or_before_each_bar():
    t = pd.date_range("2025-01-01", periods=4, freq="h", tz="UTC")
    asset = pd.Series([100, 101, 102, 103.0], index=t)
    prob = pd.Series([0.40, 0.50, 0.60], index=[t[0] - pd.Timedelta(minutes=5),
                                               t[1] + pd.Timedelta(minutes=30),   # after bar 1 closes
                                               t[3]])
    df = align(prob, asset)
    # bar 1 must still see 0.40: the 0.50 print happened 30 minutes later
    assert df.loc[t[1], "prob"] == 0.40
    assert df.loc[t[2], "prob"] == 0.50
    assert df.loc[t[3], "prob"] == 0.60


def test_shocks_respect_cooldown_and_use_past_volatility(lagged):
    cfg = Config()
    ev = detect_shocks(lagged, cfg)
    assert len(ev) > 30
    gaps = ev["time"].diff().dropna()
    assert (gaps >= pd.Timedelta(hours=cfg.cooldown_hours)).all()
    assert (ev["z"].abs() > cfg.shock_z).all()


def test_sensitivity_recovers_true_beta(lagged):
    cfg = Config()
    beta = fit_sensitivity(lagged, detect_shocks(lagged, cfg), cfg.response_horizon)
    assert 11 < beta < 19


def test_event_study_detects_lag_and_its_absence(lagged, no_lag):
    cfg = Config()
    for df, expect_lag in [(lagged, True), (no_lag, False)]:
        paths = event_paths(df, detect_shocks(df, cfg), cfg.pre_hours, cfg.post_hours)["prob"]
        prof = adjustment_profile(paths.mean(), cfg.response_horizon)
        if expect_lag:
            assert prof["share_on_shock_bar"] < 0.5
            assert prof["post_shock_drift"] > 0.01
        else:
            assert prof["share_on_shock_bar"] > 0.8


def test_lead_lag_points_from_asset_to_probability(lagged):
    ccf = cross_correlation(lagged, 4)
    assert ccf.loc[1] > 0.05 and abs(ccf.loc[-1]) < 0.03
    regs = lead_lag_regressions(lagged)
    assert regs["prob_on_asset"].loc["r_lag1", "t"] > 5


def test_lagged_market_is_tradable_and_instant_market_is_not(lagged, no_lag):
    cfg = Config()
    results = {}
    for name, df in [("lagged", lagged), ("no_lag", no_lag)]:
        ev = detect_shocks(df, cfg)
        beta = fit_sensitivity(df, ev, cfg.response_horizon)
        results[name] = trade_events(df, ev, beta, cfg, delay=0)
    assert len(results["lagged"]) > 10 and results["lagged"]["pnl"].mean() > 0.02
    # Without a lag the market already sits at the implied level, so the gap rule rarely fires.
    assert len(results["no_lag"]) < len(results["lagged"]) / 3
