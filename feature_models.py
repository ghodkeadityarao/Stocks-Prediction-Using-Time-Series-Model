"""Price forecasts from models that also see market information, tested like every other model.

    python feature_models.py

A ridge regression and a gradient-boosting model read the features built in market.py (the stock's
own recent returns and volatility, its volume, the market index and the fear index) and predict
the return over the next 1, 5 and 20 days. The price forecast is the last close times the
exponential of that return.

No look-ahead: each training example's outcome has already happened by the forecast date, and the
models are refitted every 10 forecasts (50 trading days).
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge

import common
import market

LONGEST = max(common.HORIZONS)
REFIT_EVERY = 10
MODELS = {
    "Ridge with market information": lambda: Ridge(alpha=1000.0),
    "Gradient boosting with market information": lambda: HistGradientBoostingRegressor(
        max_depth=3, learning_rate=0.05, max_iter=80, min_samples_leaf=40, random_state=0),
}


def forecaster(features, make_model):
    """One model per horizon, predicting the log return over that horizon."""
    state = {"calls": 0, "fits": {}}
    X = features.to_numpy(dtype=float)

    def forecast(hist, future_dates):
        t = len(hist) - 1
        lp = np.log(hist.to_numpy())
        ok = np.isfinite(X[: t + 1]).all(axis=1)
        if state["calls"] % REFIT_EVERY == 0:
            for h in common.HORIZONS:
                rows = np.flatnonzero(ok[: t - h + 1])      # only examples whose outcome is known
                y = lp[rows + h] - lp[rows]
                scale = y.std()
                state["fits"][h] = (make_model().fit(X[rows], y / scale), scale)
        state["calls"] += 1
        cum = [state["fits"][h][0].predict(X[t:t + 1])[0] * state["fits"][h][1] for h in common.HORIZONS]
        path = np.interp(np.arange(1, LONGEST + 1), [0, *common.HORIZONS], [0.0, *cum])
        return hist.iloc[-1] * np.exp(path)
    return forecast


def run(name):
    prices = common.load_prices(name)
    features = market.feature_frame(name, prices)
    return prices, {label: common.backtest(prices, forecaster(features, make))
                    for label, make in MODELS.items()}


if __name__ == "__main__":
    import sys
    pd.set_option("display.width", 250)
    for name in (sys.argv[1:] or common.TICKERS):
        prices, out = run(name)
        out["Naive"] = common.backtest(prices, common.naive)
        print(f"\n{name}")
        print(common.compare_to_naive(out).round(3).to_string(index=False))
