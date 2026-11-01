"""Prophet price forecasts, tested with the same walk-forward backtest as the other models.

    python prophet_model.py

Prophet is fitted to the log price with a yearly seasonal pattern. It is run twice: with its
default settings, and with trend settings chosen on the year before the test window (so the test
days are never used to choose them). Only the dates of the days to forecast are given to it,
never their prices.
(The file is not called prophet.py because that would hide the prophet package.)
"""
import itertools
import logging
from functools import partial

import numpy as np
import pandas as pd
from prophet import Prophet

import common

logging.getLogger("cmdstanpy").setLevel(logging.ERROR)
logging.getLogger("prophet").setLevel(logging.ERROR)


DEFAULT = {"changepoint_range": 0.8, "changepoint_prior_scale": 0.05}
GRID = [{"changepoint_range": r, "changepoint_prior_scale": c}
        for r, c in itertools.product((0.8, 0.95), (0.05, 0.5))]


def prophet_forecaster(hist, future_dates, config=DEFAULT):
    frame = pd.DataFrame({"ds": hist.index, "y": np.log(hist.to_numpy())})
    model = Prophet(daily_seasonality=False, weekly_seasonality=False, yearly_seasonality=True, **config)
    model.fit(frame)
    predicted = model.predict(pd.DataFrame({"ds": future_dates}))
    return np.exp(predicted["yhat"].to_numpy())


def select_config(train_prices):
    """Pick the trend settings with the lowest error over the last year before the test window."""
    scores = []
    for config in GRID:
        df = common.backtest(train_prices, partial(prophet_forecaster, config=config), n_test=252, step=21)
        scores.append((sum(m["rmse_pct"] for m in common.forecast_metrics(df).values()), config))
    return min(scores, key=lambda x: x[0])[1]


def run(name):
    prices = common.load_prices(name)
    config = select_config(prices.iloc[: common.split_point(prices)])
    return prices, config, {
        "Naive": common.backtest(prices, common.naive),
        "Prophet (default settings)": common.backtest(prices, prophet_forecaster),
        "Prophet (settings chosen before the test window)": common.backtest(
            prices, partial(prophet_forecaster, config=config)),
    }


if __name__ == "__main__":
    for name in common.TICKERS:
        prices, config, forecasts = run(name)
        table = common.compare_to_naive(forecasts)
        print(f"\n{name}: chosen settings {config}")
        print(table.round(3).to_string(index=False))
