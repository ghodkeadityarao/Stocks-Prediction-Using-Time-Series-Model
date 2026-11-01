"""SARIMA price forecasts with a real seasonal period, tested against ARIMA and naive baselines.

    python sarima.py

The seasonal period is chosen from the data: weekly (5 trading days) or monthly (21 trading days),
whichever lowers the AIC most on the days before the test window.
"""

import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX

import arima
import common

PERIODS = (5, 21)
REFIT_EVERY = 10   # re-estimate the parameters every 10 origins; the data is refiltered at every origin


def _model(y, order, season):
    seasonal = (1, 0, 1, season) if season else (0, 0, 0, 0)
    return SARIMAX(y, order=order, seasonal_order=seasonal, trend="t")


def select_season(train_prices, order):
    """AIC of the non-seasonal model and of each seasonal candidate (training days only)."""
    y = np.log(train_prices.to_numpy())
    aic = {"none": float(_model(y, order, None).fit(disp=False).aic)}
    for s in PERIODS:
        aic[str(s)] = float(_model(y, order, s).fit(disp=False, maxiter=200).aic)
    best = min(PERIODS, key=lambda s: aic[str(s)])
    return best, aic


def sarima_forecaster(order, season):
    state = {"calls": 0, "params": None}

    def forecast(hist, future_dates):
        mod = _model(np.log(hist.to_numpy()), order, season)
        if state["calls"] % REFIT_EVERY == 0:
            res = mod.fit(start_params=state["params"], disp=False, maxiter=200)
            state["params"] = res.params
        else:
            res = mod.filter(state["params"])
        state["calls"] += 1
        return np.exp(res.forecast(len(future_dates)))
    return forecast


def run(name):
    prices = common.load_prices(name)
    train = prices.iloc[: common.split_point(prices)]
    order = arima.select_order(train)
    season, aic = select_season(train, order)
    forecasts = {
        "Naive": common.backtest(prices, common.naive),
        f"ARIMA{order}": common.backtest(prices, arima.arima_forecaster(order)),
        f"SARIMA{order}(1,0,1,{season})": common.backtest(prices, sarima_forecaster(order, season)),
    }
    return prices, order, season, aic, forecasts


if __name__ == "__main__":
    for name in common.TICKERS:
        prices, order, season, aic, forecasts = run(name)
        table = common.compare_to_naive(forecasts)
        print(f"\n{name}: order {order}, seasonal period {season}, AIC by seasonal period "
              f"{ {k: round(v, 1) for k, v in aic.items()} }")
        print(table.round(3).to_string(index=False))
