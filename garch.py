"""GARCH volatility forecasts, tested against simple volatility baselines.

    python garch.py

GARCH describes how volatile a price is, not where it is going, so it is judged on volatility:
at each origin the model forecasts the average daily volatility over the next 1, 5 and 20 trading
days, and that is compared with the volatility that actually followed (root mean square of the
daily returns over those days). GARCH is fitted to daily percentage returns.
"""

import numpy as np
import pandas as pd
from arch import arch_model

import common

LONGEST = max(common.HORIZONS)
MODELS = {
    "GARCH(1,1) normal": dict(vol="GARCH", p=1, q=1, dist="normal"),
    "GARCH(1,1) Student-t": dict(vol="GARCH", p=1, q=1, dist="t"),
    "GJR-GARCH(1,1) Student-t": dict(vol="GARCH", p=1, o=1, q=1, dist="t"),
}
BASELINE = "Rolling 21-day"


def garch_forecaster(**spec):
    """Variance forecast for the next LONGEST days, refitted at every origin."""
    def forecast(returns):
        res = arch_model(returns, mean="Constant", **spec).fit(disp="off", update_freq=0)
        return res.forecast(horizon=LONGEST, reindex=False).variance.iloc[-1].to_numpy()
    return forecast


def rolling(window):
    """Flat forecast: the volatility of the last `window` days."""
    return lambda returns: np.full(LONGEST, np.mean(returns.to_numpy()[-window:] ** 2))


def ewma(returns, lam=0.94):
    """RiskMetrics: an exponentially weighted variance, held flat."""
    r2 = returns.to_numpy() ** 2
    var = r2[:30].mean()
    for x in r2:
        var = lam * var + (1 - lam) * x
    return np.full(LONGEST, var)


def vol_backtest(prices, forecaster):
    """One row per origin and horizon: forecast volatility and the volatility that followed."""
    returns = 100 * np.log(prices).diff().dropna()
    n = len(prices)
    rows = []
    for t in common.origins(n):
        hist = returns.loc[: prices.index[t]]
        path = np.asarray(forecaster(hist), dtype=float)
        for h in common.HORIZONS:
            if t + h <= n - 1:
                realized = returns.loc[prices.index[t + 1]: prices.index[t + h]]
                rows.append((prices.index[t], h, float(np.sqrt(path[:h].mean())),
                             float(np.sqrt(np.mean(realized.to_numpy() ** 2)))))
    return pd.DataFrame(rows, columns=["origin", "h", "forecast", "realized"])


def _qlike(df):
    """Lower is better. Punishes an under-forecast of volatility more than an over-forecast."""
    return np.log(df["forecast"] ** 2) + df["realized"] ** 2 / df["forecast"] ** 2


def path_from_averages(v1, v5, v20):
    """A 20-day variance path whose averages over the first 1, 5 and 20 days are v1, v5 and v20."""
    path = np.empty(LONGEST)
    path[0] = v1
    path[1:5] = max((5 * v5 - v1) / 4, 1e-6)
    path[5:] = max((20 * v20 - 5 * v5) / 15, 1e-6)
    return path


def compare_to_baseline(forecasts, base=BASELINE):
    rows = []
    for name, df in forecasts.items():
        for h, g in df.groupby("h"):
            b = forecasts[base][forecasts[base]["h"] == h].set_index("origin").loc[g["origin"]]
            q, qb = _qlike(g).to_numpy(), _qlike(b.reset_index()).to_numpy()
            rows.append({
                "model": name, "horizon_days": h, "origins": len(g),
                "rmse_vol_pct": float(np.sqrt(np.mean((g["forecast"] - g["realized"]) ** 2))),
                "mae_vol_pct": float(np.mean(np.abs(g["forecast"] - g["realized"]))),
                "correlation": float(np.corrcoef(g["forecast"], g["realized"])[0, 1]),
                "qlike": float(q.mean()),
                "rmse_vs_baseline": float(np.sqrt(np.mean((g["forecast"] - g["realized"]) ** 2))
                                          / np.sqrt(np.mean((b["forecast"].to_numpy() - b["realized"].to_numpy()) ** 2))),
                "qlike_gain_vs_baseline": float(qb.mean() - q.mean()),
                "dm_p_value": float("nan") if name == base else common.diebold_mariano(q, qb, h),
            })
            # a real gain needs a lower QLIKE (not just a lower RMSE) and a significant test
            rows[-1]["significantly_better"] = bool(name != base and rows[-1]["qlike_gain_vs_baseline"] > 0
                                                    and rows[-1]["dm_p_value"] < 0.05)
    return pd.DataFrame(rows)


def compare_pair(forecasts, model, other):
    """Head to head: does `model` beat `other` (not just the rolling baseline)? Same QLIKE test."""
    rows = []
    for h in common.HORIZONS:
        a = forecasts[model][forecasts[model]["h"] == h].reset_index(drop=True)
        b = forecasts[other][forecasts[other]["h"] == h].reset_index(drop=True)
        qa, qb = _qlike(a).to_numpy(), _qlike(b).to_numpy()
        rows.append({
            "model": model, "versus": other, "horizon_days": h,
            "rmse_ratio": float(np.sqrt(np.mean((a["forecast"] - a["realized"]) ** 2))
                                / np.sqrt(np.mean((b["forecast"] - b["realized"]) ** 2))),
            "qlike_gain": float(qb.mean() - qa.mean()),
            "dm_p_value": common.diebold_mariano(qa, qb, h),
        })
    return pd.DataFrame(rows)


def run(name):
    prices = common.load_prices(name)
    forecasters = {BASELINE: rolling(21), "Rolling 63-day": rolling(63), "EWMA (0.94)": ewma}
    forecasters.update({k: garch_forecaster(**v) for k, v in MODELS.items()})
    return prices, {k: vol_backtest(prices, f) for k, f in forecasters.items()}


if __name__ == "__main__":
    for name in common.TICKERS:
        prices, forecasts = run(name)
        table = compare_to_baseline(forecasts)
        print(f"\n{name}")
        print(table.round(3).to_string(index=False))
