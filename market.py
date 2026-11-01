"""Market information that goes with every stock: the stock's trading volume, its market index
(Nifty 50 for the Indian stocks, S&P 500 for the US ones) and the matching fear index
(India VIX or VIX). All of it is known at the close of each day, and gaps are filled with the
previous value, so it never uses the future.

The same features are built for every stock:
  own history   the last 5 daily returns, 5- and 20-day average return, 5- and 20-day volatility,
                the 60-day return
  volume        today's volume relative to its 20-day average (log)
  market        the market's return today and yesterday, its 21-day volatility, the log of the fear
                index and its daily change
"""
import numpy as np
import pandas as pd
import yfinance as yf

import common

MARKET_COLUMNS = ["mkt_r0", "mkt_r1", "mkt_vol21", "fear", "dfear"]


def _history(symbol):
    d = yf.Ticker(symbol).history(start=common.START, end=common.END)
    d.index = pd.to_datetime(d.index.date)
    d.index.name = "Date"
    return d


def _cached(path, build):
    if not path.exists():
        common.DATA.mkdir(exist_ok=True)
        build().to_csv(path)
    return pd.read_csv(path, index_col="Date", parse_dates=True)


def load_market(market):
    """Daily index and fear-index closes for "IN" or "US"."""
    symbols = common.MARKETS[market]

    def build():
        out = pd.DataFrame({"index": _history(symbols["index"])["Close"],
                            "fear": _history(symbols["fear"])["Close"]})
        return out

    return _cached(common.DATA / f"market_{market}.csv", build)


def load_volume(name):
    return _cached(common.DATA / f"volume_{name}.csv",
                   lambda: _history(common.TICKERS[name])[["Volume"]])["Volume"]


def market_frame(name, index):
    """The market columns, lined up with the stock's trading days."""
    m = load_market(common.market_of(name)).reindex(index).ffill()
    r = np.log(m["index"]).diff()
    return pd.DataFrame({"mkt_r0": r, "mkt_r1": r.shift(1), "mkt_vol21": r.rolling(21).std(),
                         "fear": np.log(m["fear"]), "dfear": np.log(m["fear"]).diff()}, index=index)


def feature_frame(name, prices):
    """Own-history, volume and market features for every day of `prices`."""
    r = np.log(prices).diff()
    f = pd.DataFrame(index=prices.index)
    for k in range(5):
        f[f"r_lag{k}"] = r.shift(k)
    f["mean5"], f["mean20"] = r.rolling(5).mean(), r.rolling(20).mean()
    f["vol5"], f["vol20"] = r.rolling(5).std(), r.rolling(20).std()
    f["mom60"] = np.log(prices / prices.shift(60))
    vol = load_volume(name).reindex(prices.index).where(lambda v: v > 0).ffill()   # a zero is missing data
    f["volu"] = np.log(vol / vol.rolling(20).mean())
    return pd.concat([f, market_frame(name, prices.index)], axis=1)
