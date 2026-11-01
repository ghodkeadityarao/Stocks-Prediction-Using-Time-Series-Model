"""Tests for the rules every result depends on: no look-ahead, correct alignment, and the metrics.

They use synthetic prices (no downloads) and run in a few seconds.
"""
import numpy as np
import pandas as pd
import pytest

import arima
import common
import garch
import market
import neural
import vol_target


@pytest.fixture(scope="module")
def prices():
    rng = np.random.default_rng(0)
    idx = pd.bdate_range("2015-01-01", periods=900)
    return pd.Series(100 * np.exp(np.cumsum(rng.normal(0.0003, 0.015, len(idx)))), index=idx)


def test_backtest_rows_match_the_real_prices(prices):
    df = common.backtest(prices, common.naive, n_test=100, step=5)
    pos = {d: i for i, d in enumerate(prices.index)}
    for r in df.itertuples():
        i = pos[r.origin]
        assert r.last == prices.iloc[i] and r.actual == prices.iloc[i + r.h]
        assert r.forecast == r.last                      # the naive forecast repeats the last close
    assert set(df.h) == set(common.HORIZONS)


def test_forecasters_only_see_data_up_to_the_origin(prices):
    """Changing prices after the origin must not change the forecast made at the origin."""
    changed = prices.copy()
    changed.iloc[-60:] *= 3.0
    f = arima.arima_forecaster((1, 1, 0))
    a = common.backtest(prices, f, n_test=100, step=10)
    b = common.backtest(changed, arima.arima_forecaster((1, 1, 0)), n_test=100, step=10)
    safe = a.origin <= prices.index[-61]                  # origins whose information excludes the change
    assert safe.sum() > 3
    assert np.allclose(a.loc[safe, "forecast"], b.loc[safe, "forecast"])


def test_the_naive_baseline_has_an_error_ratio_of_one(prices):
    df = common.backtest(prices, common.naive, n_test=100, step=5)
    table = common.compare_to_naive({"Naive": df, "Other": df.copy()})
    assert np.allclose(table[table.model == "Other"].rmse_vs_naive, 1.0)


def test_diebold_mariano_separates_real_gains_from_noise():
    rng = np.random.default_rng(1)
    base = rng.normal(1.0, 0.1, 200)
    assert common.diebold_mariano(base - 0.3, base, 1) < 0.001      # consistently lower loss
    assert common.diebold_mariano(base + rng.normal(0, 0.1, 200), base, 1) > 0.01


def test_variance_path_has_the_requested_averages():
    path = garch.path_from_averages(1.0, 1.2, 1.5)
    assert np.isclose(path[:1].mean(), 1.0) and np.isclose(path[:5].mean(), 1.2) and np.isclose(path[:20].mean(), 1.5)


def test_volatility_backtest_scores_against_what_followed(prices):
    df = garch.vol_backtest(prices, garch.rolling(21))
    r = 100 * np.log(prices).diff()
    row = df[df.h == 5].iloc[3]
    i = list(prices.index).index(row.origin)
    assert np.isclose(row.realized, np.sqrt(np.mean(r.iloc[i + 1:i + 6].to_numpy() ** 2)))
    assert np.isclose(row.forecast, np.sqrt(np.mean(r.iloc[i - 20:i + 1].to_numpy() ** 2)))


def test_networks_only_train_on_outcomes_that_already_happened():
    r = np.arange(200.0)
    ends, _ = neural._samples(r, lambda w, e: w[:, None])
    assert ends.max() + max(common.HORIZONS) <= len(r)


def test_market_features_do_not_use_the_future(prices, monkeypatch):
    rng = np.random.default_rng(2)
    idx = prices.index
    vol = pd.Series(rng.integers(1_000, 5_000, len(idx)).astype(float), index=idx)
    mk = pd.DataFrame({"index": 1000 * np.exp(np.cumsum(rng.normal(0, 0.01, len(idx)))),
                       "fear": 15 + rng.normal(0, 1, len(idx))}, index=idx)
    monkeypatch.setattr(market, "load_volume", lambda name: vol)
    monkeypatch.setattr(market, "load_market", lambda m: mk)
    monkeypatch.setattr(common, "TICKERS", {**common.TICKERS, "FAKE": "FAKE.NS"})
    base = market.feature_frame("FAKE", prices)
    vol2, mk2, p2 = vol.copy(), mk.copy(), prices.copy()
    vol2.iloc[-50:] *= 10
    mk2.iloc[-50:] *= 2
    p2.iloc[-50:] *= 2
    monkeypatch.setattr(market, "load_volume", lambda name: vol2)
    monkeypatch.setattr(market, "load_market", lambda m: mk2)
    changed = market.feature_frame("FAKE", p2)
    pd.testing.assert_frame_equal(base.iloc[:-50], changed.iloc[:-50])


def test_volatility_targeting_never_levers_up(prices):
    rng = np.random.default_rng(3)
    origins = prices.index[300:600:5]
    df = pd.DataFrame({"origin": origins, "h": vol_target.H, "forecast": rng.uniform(0.2, 4.0, len(origins)),
                       "realized": 1.0})
    held, exposure, firsts = vol_target._held(prices, df, 0.15)
    assert exposure.max() <= vol_target.CAP + 1e-12
    daily_target = 100 * 0.15 / np.sqrt(vol_target.DAYS)
    expected = np.minimum(1.0, daily_target / df.forecast.to_numpy())
    assert np.allclose(exposure, expected)
