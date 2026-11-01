"""Shared pieces for every model: price data, the walk-forward test, baselines and metrics.

All price models are tested the same way. A forecast is made from the data up to one day (the
origin), for 1, 5 and 20 trading days ahead, and compared with what the price actually did.
Origins run through the last two years of the data, one every five trading days, and every model
only ever sees data up to its origin.
"""
import math
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore", message=".*date index.*")
warnings.filterwarnings("ignore", message=".*inplace keyword.*")
warnings.filterwarnings("ignore", message=".*adfuller currently returns.*")
warnings.filterwarnings("ignore", message=".*Non-invertible starting.*")
warnings.filterwarnings("ignore", message=".*Non-stationary starting.*")
warnings.filterwarnings("ignore", message=".*Maximum Likelihood optimization failed.*")

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
RESULTS = ROOT / "results"

TICKERS = {"PERSISTENT": "PERSISTENT.NS", "TECHM": "TECHM.NS", "INFOSYS": "INFY.NS", "TCS": "TCS.NS",
           "HDFCBANK": "HDFCBANK.NS", "RELIANCE": "RELIANCE.NS", "AAPL": "AAPL", "MSFT": "MSFT"}
# the market index and fear index (volatility index) that go with each stock
MARKETS = {"IN": {"index": "^NSEI", "fear": "^INDIAVIX"}, "US": {"index": "^GSPC", "fear": "^VIX"}}


def market_of(name):
    return "IN" if TICKERS[name].endswith(".NS") else "US"
START, END = "2013-09-02", "2023-09-01"   # END is exclusive; adjusted closing prices
N_TEST = 504                              # last two years of trading days
STEP = 5                                  # a new forecast origin every 5 trading days
HORIZONS = (1, 5, 20)


def load_prices(name):
    """Daily adjusted closing prices. Downloaded once, then read from data/<name>.csv."""
    path = DATA / f"{name}.csv"
    if not path.exists():
        import yfinance as yf

        DATA.mkdir(exist_ok=True)
        close = yf.Ticker(TICKERS[name]).history(start=START, end=END)["Close"]
        close.index = pd.to_datetime(close.index.date)
        close.index.name = "Date"
        close.to_csv(path, header=["Close"])
    return pd.read_csv(path, index_col="Date", parse_dates=True)["Close"]


def split_point(prices, n_test=N_TEST):
    """Number of days before the test window. Anything chosen once (such as a model order)
    uses only these days."""
    return len(prices) - n_test


def origins(n, n_test=N_TEST, step=STEP):
    """Positions of the last day of information. Their forecasts land inside the test window."""
    return range(n - n_test - 1, n - 1, step)


# --- baselines -------------------------------------------------------------------------------

def naive(hist, future_dates):
    """Tomorrow equals today: repeat the last close."""
    return np.full(len(future_dates), hist.iloc[-1])


def drift(hist, future_dates):
    """Last close plus the average daily growth seen so far."""
    lp = np.log(hist.to_numpy())
    mu = (lp[-1] - lp[0]) / (len(lp) - 1)
    return np.exp(lp[-1] + mu * np.arange(1, len(future_dates) + 1))


# --- walk-forward test -----------------------------------------------------------------------

def backtest(prices, forecaster, n_test=N_TEST, step=STEP):
    """Run `forecaster(history, future_dates) -> prices for those dates` from every origin.

    Only the dates of the future days are passed in (never their prices).
    """
    longest = max(HORIZONS)
    n = len(prices)
    rows = []
    for t in origins(n, n_test, step):
        hist = prices.iloc[: t + 1]
        future_dates = prices.index[t + 1: t + 1 + longest]
        forecast = np.asarray(forecaster(hist, future_dates), dtype=float)
        for h in HORIZONS:
            if t + h <= n - 1 and h <= len(forecast):
                rows.append((prices.index[t], h, prices.iloc[t], prices.iloc[t + h], forecast[h - 1]))
    return pd.DataFrame(rows, columns=["origin", "h", "last", "actual", "forecast"])


def _log_error(df):
    return np.log(df["actual"] / df["forecast"].clip(lower=1e-6))


def forecast_metrics(df):
    """Errors per horizon: RMSE in the stock's own currency, RMSE and MAPE as a percent of the price, and how often
    the forecast gets the direction of the move right."""
    out = {}
    for h, g in df.groupby("h"):
        change = np.sign(g["actual"] - g["last"])
        called = np.sign(g["forecast"] - g["last"])
        valid = (change != 0) & (called != 0)
        out[h] = {
            "rmse_price": float(np.sqrt(np.mean((g["actual"] - g["forecast"]) ** 2))),
            "rmse_pct": float(100 * np.sqrt(np.mean(_log_error(g) ** 2))),
            "mape_pct": float(100 * np.mean(np.abs(g["actual"] - g["forecast"]) / g["actual"])),
            "direction_acc": float((change[valid] == called[valid]).mean()) if valid.any() else float("nan"),
            "origins": int(len(g)),
        }
    return out


def diebold_mariano(loss_model, loss_base, h):
    """p-value of a two-sided Diebold-Mariano test (Newey-West variance) on two loss series.

    Overlapping forecasts make the losses correlated, so the variance allows for that.
    """
    d = np.asarray(loss_model, float) - np.asarray(loss_base, float)
    n = len(d)
    lag = max(math.ceil(h / STEP) - 1, 0)
    dc = d - d.mean()
    var = np.sum(dc * dc) / n
    for k in range(1, lag + 1):
        var += 2 * (1 - k / (lag + 1)) * np.sum(dc[k:] * dc[:-k]) / n
    if var <= 0:
        return float("nan")
    stat = d.mean() / math.sqrt(var / n)
    return float(2 * stats.t.sf(abs(stat), df=n - 1))


def compare_to_naive(forecasts, base="Naive"):
    """One row per model and horizon, with its error relative to the naive baseline."""
    rows = []
    base_df = forecasts[base]
    for name, df in forecasts.items():
        metrics = forecast_metrics(df)
        for h, m in metrics.items():
            row = {"model": name, "horizon_days": h, **m}
            b = base_df[base_df["h"] == h].set_index("origin")
            g = df[df["h"] == h].set_index("origin")
            b = b.loc[g.index]
            row["rmse_vs_naive"] = row["rmse_price"] / float(np.sqrt(np.mean((b["actual"] - b["forecast"]) ** 2)))
            row["dm_p_value"] = (float("nan") if name == base else
                                 diebold_mariano(_log_error(g) ** 2, _log_error(b) ** 2, h))
            rows.append(row)
    return pd.DataFrame(rows)
