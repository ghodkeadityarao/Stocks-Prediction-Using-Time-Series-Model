"""Volatility targeting: size the position with a volatility forecast and compare with buy-and-hold.

    python vol_target.py

Every 5 trading days (at each forecast origin) the position is set to
    exposure = min(1, target volatility / forecast volatility)
and held for the next 5 trading days, with the rest in cash at zero return. So the position is
cut when the forecast says the stock is about to be more volatile than usual, and it is never
leveraged. The target is a fixed annualised volatility (15% is the main one; 10% and 20% are
reported too, so no target is picked after the fact).

The test window, the forecasts and the no-look-ahead rule are the same as in the volatility
forecast test. Costs are 10 basis points on every unit of exposure traded, and buy-and-hold pays
the same cost to buy in. Two controls keep the comparison honest: buy-and-hold, and a static
position held at the same average exposure as the volatility-targeted one, which shows whether
any benefit comes from the forecast or just from holding less.

Run on its own, this file tests the GARCH-family and rolling-window forecasts; run_all.py runs it
on every volatility model.
"""
import numpy as np
import pandas as pd

import common
import garch

H = 5                  # forecast horizon used to size the position, and days each position is held
CAP = 1.0              # no leverage
COST = 0.001           # 10 basis points per unit of exposure traded
DAYS = 252
TARGETS = (0.10, 0.15, 0.20)   # annualised volatility targets; 0.15 is the main one
MAIN_TARGET = 0.15
BLOCK = 10             # trading days per block in the bootstrap
RESAMPLES = 2000


def _held(prices, forecast_df, target_annual):
    """Daily returns of the volatility-targeted position before costs, with its exposures."""
    target = 100 * target_annual / np.sqrt(DAYS)          # daily volatility in percent
    simple = prices.pct_change()
    pos = {d: i for i, d in enumerate(prices.index)}
    rows = forecast_df[forecast_df["h"] == H].sort_values("origin")
    exposure = np.minimum(CAP, target / rows["forecast"].to_numpy())
    pieces, firsts = [], []
    for origin, e in zip(rows["origin"], exposure):
        i = pos[origin]
        pieces.append(e * simple.iloc[i + 1: i + 1 + H])
        firsts.append(prices.index[i + 1])
    return pd.concat(pieces), exposure, firsts


def _net(returns, exposure, firsts):
    """Subtract the trading cost on the first day of each holding period."""
    net = returns.copy()
    prev = 0.0
    for e, day in zip(exposure, firsts):
        net.loc[day] -= COST * abs(e - prev)
        prev = e
    return net


def metrics(daily, exposure):
    equity = (1 + daily).cumprod()
    n = len(daily)
    return {
        "annual_return": float(equity.iloc[-1] ** (DAYS / n) - 1),
        "annual_volatility": float(daily.std() * np.sqrt(DAYS)),
        "sharpe": float(daily.mean() / daily.std() * np.sqrt(DAYS)),
        "max_drawdown": float((equity / equity.cummax() - 1).min()),
        "average_exposure": float(np.mean(exposure)),
        "annual_turnover": float(np.sum(np.abs(np.diff(np.r_[0.0, exposure]))) / (n / DAYS)),
    }


def _sharpe(x):
    return x.mean(axis=-1) / x.std(axis=-1) * np.sqrt(DAYS)


def sharpe_gain_interval(strategy, control, seed=0):
    """95% interval for (Sharpe of strategy - Sharpe of control), resampling 10-day blocks of days."""
    a, b = strategy.to_numpy(), control.to_numpy()
    n = len(a)
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, n - BLOCK, size=(RESAMPLES, n // BLOCK + 1))
    idx = (starts[:, :, None] + np.arange(BLOCK)).reshape(RESAMPLES, -1)[:, :n]
    gain = _sharpe(a[idx]) - _sharpe(b[idx])
    return float(np.percentile(gain, 2.5)), float(np.percentile(gain, 97.5))


def run(prices, vol_forecasts, target_annual=MAIN_TARGET):
    """One row per strategy. Also returns the daily growth curves for the GARCH(1,1) strategy."""
    simple = prices.pct_change()
    held, _, _ = _held(prices, vol_forecasts[garch.BASELINE], target_annual)
    days = held.index
    bh = simple.loc[days].copy()
    bh.iloc[0] -= COST                                            # buy in once
    rows = [{"strategy": "Buy and hold", **metrics(bh, [1.0])}]
    curves = {"Buy and hold": (1 + bh).cumprod()}
    for model, df in vol_forecasts.items():
        held, exposure, firsts = _held(prices, df, target_annual)
        net = _net(held, exposure, firsts)
        static_e = float(np.mean(exposure))
        static = static_e * simple.loc[net.index]
        static.iloc[0] -= COST * static_e
        m, s = metrics(net, exposure), metrics(static, [static_e])
        low, high = sharpe_gain_interval(net, static)
        rows.append({"strategy": f"Volatility target: {model}", **m,
                     "static_sharpe": s["sharpe"], "static_max_drawdown": s["max_drawdown"],
                     "static_annual_volatility": s["annual_volatility"],
                     "sharpe_gain_vs_static_low": low, "sharpe_gain_vs_static_high": high})
        if model == "GARCH(1,1) normal":
            curves["Volatility target (GARCH)"] = (1 + net).cumprod()
            curves["Static, same average exposure"] = (1 + static).cumprod()
    return pd.DataFrame(rows).assign(target_annual_volatility=target_annual), curves


if __name__ == "__main__":
    pd.set_option("display.width", 250)
    for name in common.TICKERS:
        prices, forecasts = garch.run(name)
        table, _ = run(prices, forecasts)
        print(f"\n{name} (target {MAIN_TARGET:.0%})")
        print(table.drop(columns=["target_annual_volatility"]).round(3).to_string(index=False))
