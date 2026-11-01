"""Run every model on every stock and save the combined tables and figures in results/.

    python run_all.py

Every stock gets exactly the same models and tests. The stocks are analysed in parallel (one
process each).
"""
import json
from concurrent.futures import ProcessPoolExecutor
from functools import partial

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import arima
import common
import feature_models
import garch
import har
import market
import neural
import prophet_model
import sarima
import vol_target

FIGURES = common.RESULTS / "figures"


def own(cell):
    return f"{cell} (average of 3 seeds)"


def with_market(cell):
    return f"{cell} with market information (average of 3 seeds)"


PRICE_MODELS = ["Drift", "ARIMA (order by AIC)", "SARIMA (period by AIC)",
                "Prophet (settings chosen before the test window)", "Prophet (default settings)",
                "ARIMA(5,2,0) on the raw price", own("LSTM"), own("GRU"), *feature_models.MODELS]
VOL_MODELS = ["GARCH(1,1) normal", "GARCH(1,1) Student-t", "GJR-GARCH(1,1) Student-t", "EWMA (0.94)",
              "Rolling 63-day", har.LABELS[0], har.LABELS[1], own("LSTM"), own("GRU"),
              with_market("LSTM"), with_market("GRU")]
GARCH = "GARCH(1,1) normal"
# (model, compared with): is the model better than the one it is paired with?
HEAD_TO_HEAD = [(own("LSTM"), GARCH), (own("GRU"), GARCH), (har.LABELS[1], GARCH),
                (with_market("LSTM"), GARCH), (with_market("GRU"), GARCH),
                (har.LABELS[1], har.LABELS[0]), (with_market("LSTM"), own("LSTM")),
                (with_market("GRU"), own("GRU"))]


def price_models(name):
    prices = common.load_prices(name)
    train = prices.iloc[: common.split_point(prices)]
    order = arima.select_order(train)
    season, aic = sarima.select_season(train, order)
    config = prophet_model.select_config(train)
    forecasts = {
        "Naive": common.backtest(prices, common.naive),
        "Drift": common.backtest(prices, common.drift),
        "ARIMA (order by AIC)": common.backtest(prices, arima.arima_forecaster(order)),
        "SARIMA (period by AIC)": common.backtest(prices, sarima.sarima_forecaster(order, season)),
        "Prophet (settings chosen before the test window)": common.backtest(
            prices, partial(prophet_model.prophet_forecaster, config=config)),
        "Prophet (default settings)": common.backtest(prices, prophet_model.prophet_forecaster),
        "ARIMA(5,2,0) on the raw price": common.backtest(prices, arima.original_forecaster),
    }
    forecasts.update(feature_models.run(name)[1])
    stability = []
    for cell in neural.CELLS:
        per_seed, mean = neural.run_price(name, cell)
        forecasts[own(cell)] = mean
        stability.append(neural.seed_spread(per_seed, forecasts["Naive"], "price").assign(
            stock=name, task="price", model=cell))
    choices = {"market": common.market_of(name), "arima_order": order,
               "adf_p_values": arima.stationarity(prices), "sarima_season": season,
               "aic_by_seasonal_period": aic, "prophet_settings": config,
               "train_days": len(train), "test_days": common.N_TEST,
               "first_day": str(prices.index[0].date()), "last_day": str(prices.index[-1].date()),
               "test_start": str(prices.index[-common.N_TEST].date()),
               "price_change_in_test_window_pct": round(100 * (prices.iloc[-1] / prices.iloc[-common.N_TEST] - 1), 1)}
    return forecasts, choices, stability


def analyse(name):
    forecasts, choices, stability = price_models(name)
    prices, vol_forecasts = garch.run(name)
    vol_forecasts.update(har.run(name)[1])
    for cell in neural.CELLS:
        for label, market_info in ((own(cell), False), (with_market(cell), True)):
            per_seed, mean = neural.run_vol(name, cell, market_info=market_info)
            vol_forecasts[label] = mean
            stability.append(neural.seed_spread(per_seed, vol_forecasts[garch.BASELINE], "vol").assign(
                stock=name, task="volatility", model=label.replace(" (average of 3 seeds)", "")))
    targeting, curves = [], None
    for target in vol_target.TARGETS:
        table, c = vol_target.run(prices, vol_forecasts, target)
        targeting.append(table.assign(stock=name))
        if target == vol_target.MAIN_TARGET:
            curves = c
    head_to_head = pd.concat([garch.compare_pair(vol_forecasts, a, b) for a, b in HEAD_TO_HEAD]).assign(stock=name)
    return (name, common.compare_to_naive(forecasts).assign(stock=name), choices, vol_forecasts,
            garch.compare_to_baseline(vol_forecasts).assign(stock=name), pd.concat(stability), head_to_head,
            pd.concat(targeting), curves)


def strip_panels(df, models, value, path, xlabel, log):
    """One panel per horizon: a dot per stock for every model, and a black diamond at the median."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 0.45 * len(models) + 1.6), sharey=True)
    for ax, h in zip(axes, common.HORIZONS):
        for i, m in enumerate(models):
            v = df[(df.model == m) & (df.horizon_days == h)][value].to_numpy()
            ax.scatter(v, np.full(len(v), i) + np.linspace(-0.18, 0.18, len(v)), s=22, color="tab:blue", alpha=0.7)
            ax.scatter([np.median(v)], [i], marker="D", s=46, color="black", zorder=3)
        ax.axvline(1, color="gray", lw=1)
        if log:
            ax.set_xscale("log")
            ticks = [t for t in (1, 1.5, 2, 3, 5, 8, 12) if t <= ax.get_xlim()[1] * 1.05]
            ax.set_xticks(ticks)
            ax.set_xticklabels([f"{t:g}" for t in ticks])
            ax.minorticks_off()
        ax.set_yticks(range(len(models)))
        ax.set_yticklabels(models)
        ax.tick_params(labelleft=(h == common.HORIZONS[0]))
        ax.set_ylim(len(models) - 0.5, -0.5)
        ax.set_title(f"{h} day{'s' if h > 1 else ''} ahead")
        ax.set_xlabel(xlabel)
        ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)


def grid(n):
    rows = (n + 1) // 2
    fig, axes = plt.subplots(rows, 2, figsize=(12, 3 * rows))
    return fig, axes.ravel()


def main():
    common.RESULTS.mkdir(exist_ok=True)
    FIGURES.mkdir(exist_ok=True)
    for name in common.TICKERS:          # download once, before the parallel runs
        common.load_prices(name)
        market.feature_frame(name, common.load_prices(name))
    with ProcessPoolExecutor(max_workers=len(common.TICKERS)) as pool:
        out = list(pool.map(analyse, list(common.TICKERS)))
    price = pd.concat([o[1] for o in out], ignore_index=True)
    vol = pd.concat([o[4] for o in out], ignore_index=True)
    pd.concat([o[6] for o in out], ignore_index=True).round(4).to_csv(
        common.RESULTS / "volatility_head_to_head.csv", index=False)
    targeting = pd.concat([o[7] for o in out], ignore_index=True)
    targeting.round(4).to_csv(common.RESULTS / "vol_targeting_metrics.csv", index=False)
    stability = pd.concat([o[5] for o in out], ignore_index=True)
    stability.round(4).to_csv(common.RESULTS / "neural_seed_stability.csv", index=False)
    choices = {o[0]: o[2] for o in out}
    vol_forecasts = {o[0]: o[3] for o in out}
    price.round(4).to_csv(common.RESULTS / "price_forecast_metrics.csv", index=False)
    vol.round(4).to_csv(common.RESULTS / "volatility_forecast_metrics.csv", index=False)
    (common.RESULTS / "model_choices.json").write_text(json.dumps(choices, indent=2, default=float))

    strip_panels(price, PRICE_MODELS, "rmse_vs_naive", FIGURES / "price_models_vs_naive.png",
                 "RMSE vs naive (1 = same, log)", log=True)
    strip_panels(vol, VOL_MODELS, "rmse_vs_baseline", FIGURES / "volatility_vs_baseline.png",
                 "RMSE vs 21-day rolling (1 = same)", log=False)

    fig, axes = grid(len(common.TICKERS))
    for ax, name in zip(axes, common.TICKERS):
        vf = vol_forecasts[name]
        for model, style in ((GARCH, dict(color="tab:blue")), (har.LABELS[1], dict(color="tab:green")),
                             (garch.BASELINE, dict(color="tab:orange", alpha=0.8))):
            d = vf[model][vf[model].h == 20].set_index("origin")
            ax.plot(d.index, d["forecast"], label=model, **style)
        d = vf[garch.BASELINE]
        d = d[d.h == 20].set_index("origin")
        ax.plot(d.index, d["realized"], color="black", lw=1, label="Volatility that followed")
        ax.set_title(f"{name}: average daily volatility over the next 20 days (%)", fontsize=9)
        ax.legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(FIGURES / "volatility_forecast.png", dpi=150)

    # Volatility targeting: growth of 1 unit invested, and risk against the two controls.
    curves = {o[0]: o[8] for o in out}
    colors = {"Buy and hold": "black", "Volatility target (GARCH)": "tab:blue",
              "Static, same average exposure": "tab:orange"}
    fig, axes = grid(len(common.TICKERS))
    for ax, name in zip(axes, common.TICKERS):
        for label, series in curves[name].items():
            ax.plot(series.index, series, label=label, color=colors[label], lw=1.3)
        ax.set_title(f"{name}: growth of 1 unit invested over the test window", fontsize=9)
        ax.legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(FIGURES / "vol_targeting_growth.png", dpi=150)

    main_target = targeting[targeting.target_annual_volatility == vol_target.MAIN_TARGET]
    g = main_target[main_target.strategy == f"Volatility target: {GARCH}"].set_index("stock")
    b = main_target[main_target.strategy == "Buy and hold"].set_index("stock")
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    x = np.arange(len(common.TICKERS))
    names = list(common.TICKERS)
    for ax, col, scol, title in ((axes[0], "annual_volatility", "static_annual_volatility", "Annual volatility"),
                                 (axes[1], "max_drawdown", "static_max_drawdown", "Maximum drawdown")):
        sign = -1 if col == "max_drawdown" else 1
        ax.bar(x - 0.27, sign * b.loc[names, col], 0.27, label="Buy and hold", color="black")
        ax.bar(x, sign * g.loc[names, col], 0.27, label="Volatility target (GARCH)", color="tab:blue")
        ax.bar(x + 0.27, sign * g.loc[names, scol], 0.27, label="Static, same average exposure", color="tab:orange")
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=25, fontsize=8)
        ax.set_title(title + " (lower is better)")
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIGURES / "vol_targeting_risk.png", dpi=150)


if __name__ == "__main__":
    main()
