"""HAR-style volatility forecasts, with and without market information.

    python har.py

A HAR model predicts future volatility from volatility over three look-back spans: the last day,
the last week (5 days) and the last month (21 days). Here the volatility of a span is the log of
its average squared return, and a ridge regression turns the three numbers into the average
variance over the next 1, 5 and 20 days. The "market information" version adds the log of the fear
index (VIX or India VIX), the market index's recent volatility and its latest absolute return.

The model is refitted at every forecast using only days whose outcome is already known, and the
variance is rescaled so the forecasts are right on average over the training days (a log-scale
regression otherwise forecasts a little low).
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

import common
import garch
import market

LONGEST = max(common.HORIZONS)
EPS = 1e-8
START = 30                     # first position with a full 21-day history for every input
LABELS = ("HAR (own volatility history)", "HAR with market information")


def _logavg(sq, e, k):
    return np.log(sq[e - k:e].mean() + EPS)


def _row(sq, mk, e, use_market):
    own = [_logavg(sq, e, 1), _logavg(sq, e, 5), _logavg(sq, e, 21)]
    if not use_market:
        return own
    r0, vol21, fear = mk[e - 1]
    return own + [fear, np.log(vol21 ** 2 + EPS), abs(r0)]


def forecaster(use_market, market_columns):
    """Variance path for the next 20 days, from the daily returns (in percent) up to the origin."""
    def forecast(returns):
        r = returns.to_numpy()
        sq = r ** 2
        mk = market_columns.reindex(returns.index).to_numpy() if use_market else None
        ends = range(START, len(r) - LONGEST + 1)
        X = np.array([_row(sq, mk, e, use_market) for e in ends])
        Y = np.array([[np.log(sq[e:e + h].mean() + EPS) for h in common.HORIZONS] for e in ends])
        mu, sd = X.mean(axis=0), X.std(axis=0)
        model = Ridge(alpha=1.0).fit((X - mu) / sd, Y)
        fitted = model.predict((X - mu) / sd)
        smear = np.clip(np.exp(Y).mean(axis=0) / np.exp(fitted).mean(axis=0), 0.5, 2.0)
        now = (np.array(_row(sq, mk, len(r), use_market)) - mu) / sd
        return garch.path_from_averages(*(np.exp(model.predict(now[None])[0]) * smear))
    return forecast


def run(name):
    prices = common.load_prices(name)
    mk = market.market_frame(name, prices.index)[["mkt_r0", "mkt_vol21", "fear"]]
    return prices, {label: garch.vol_backtest(prices, forecaster(use, mk))
                    for label, use in zip(LABELS, (False, True))}


if __name__ == "__main__":
    import sys
    pd.set_option("display.width", 250)
    for name in (sys.argv[1:] or common.TICKERS):
        prices, out = run(name)
        out[garch.BASELINE] = garch.vol_backtest(prices, garch.rolling(21))
        out["GARCH(1,1) normal"] = garch.vol_backtest(prices, garch.garch_forecaster(**garch.MODELS["GARCH(1,1) normal"]))
        t = garch.compare_to_baseline(out)
        print(f"\n{name}")
        print(t[["model", "horizon_days", "rmse_vol_pct", "correlation", "qlike", "rmse_vs_baseline",
                 "dm_p_value", "significantly_better"]].round(3).to_string(index=False))
