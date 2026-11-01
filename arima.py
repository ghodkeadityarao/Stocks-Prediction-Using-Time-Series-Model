"""ARIMA price forecasts, tested with a walk-forward backtest against naive baselines.

    python arima.py
"""
import itertools

import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller

import common


def stationarity(prices):
    """ADF p-values for the log price and for its first difference (is one difference enough?)."""
    lp = np.log(prices.to_numpy())
    return {"log_price": float(adfuller(lp)[1]), "first_difference": float(adfuller(np.diff(lp))[1])}


def select_order(train_prices, max_p=3, max_q=3):
    """Pick (p, 1, q) by AIC using only the days before the test window."""
    y = np.log(train_prices.to_numpy())
    best = None
    for p, q in itertools.product(range(max_p + 1), range(max_q + 1)):
        try:
            aic = ARIMA(y, order=(p, 1, q), trend="t").fit().aic
        except Exception:
            continue
        if best is None or aic < best[0]:
            best = (aic, (p, 1, q))
    return best[1]


def arima_forecaster(order):
    """ARIMA on the log price with a drift term, refitted at every origin."""
    def forecast(hist, future_dates):
        res = ARIMA(np.log(hist.to_numpy()), order=order, trend="t").fit()
        return np.exp(res.forecast(len(future_dates)))
    return forecast


def original_forecaster(hist, future_dates):
    """ARIMA(5,2,0) fitted to the raw price: two differences and no log."""
    res = ARIMA(hist.to_numpy(), order=(5, 2, 0)).fit()
    return res.forecast(len(future_dates))


def run(name):
    prices = common.load_prices(name)
    order = select_order(prices.iloc[: common.split_point(prices)])
    forecasts = {
        "Naive": common.backtest(prices, common.naive),
        "Drift": common.backtest(prices, common.drift),
        f"ARIMA{order}": common.backtest(prices, arima_forecaster(order)),
        "ARIMA(5,2,0) on the raw price": common.backtest(prices, original_forecaster),
    }
    return prices, order, forecasts


if __name__ == "__main__":
    for name in common.TICKERS:
        prices, order, forecasts = run(name)
        table = common.compare_to_naive(forecasts)
        print(f"\n{name}: order {order}, ADF p-values {stationarity(prices)}")
        print(table.round(3).to_string(index=False))
