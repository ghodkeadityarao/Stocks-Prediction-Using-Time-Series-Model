"""LSTM and GRU forecasts of price and volatility, tested with the same walk-forward backtest.

    python neural.py

Both networks read the last 30 daily returns and predict three things at once: the return over
the next 1, 5 and 20 days (price task) or the volatility over those horizons (volatility task).
The volatility networks are also run with market information (the market index's daily return and
the log of the fear index over the same 30 days) as two extra inputs.

No look-ahead: at every origin the scaling uses only the days up to that origin, every training
sample's target has already happened by then, and the epoch to keep is chosen on the most recent
15% of the training samples. A network is retrained every REFIT_EVERY origins (50 trading days).
Each model is trained from three random seeds; the forecast is their average, and the spread
between seeds is kept so a lucky seed cannot pass for an improvement.
"""
import numpy as np
import pandas as pd
import torch
from torch import nn

import common
import garch
import market

torch.set_num_threads(1)

HORIZONS = common.HORIZONS            # (1, 5, 20)
LONGEST = max(HORIZONS)
WINDOW = 30
HIDDEN = 16
EPOCHS = 30
REFIT_EVERY = 10                      # origins, i.e. every 50 trading days
SEEDS = (0, 1, 2)
CELLS = ("LSTM", "GRU")


class Net(nn.Module):
    def __init__(self, cell, n_in):
        super().__init__()
        self.rnn = getattr(nn, cell)(n_in, HIDDEN, batch_first=True)
        self.out = nn.Linear(HIDDEN, len(HORIZONS))

    def forward(self, x):
        return self.out(self.rnn(x)[0][:, -1])


def _samples(r, features):
    """Windows of past returns and, for each, what happened over the next 1, 5 and 20 days.

    Only windows whose full 20-day future is already inside `r` are used.
    """
    ends = range(WINDOW, len(r) - LONGEST + 1)
    x = np.stack([features(r[e - WINDOW:e], e) for e in ends])
    return np.asarray(list(ends)), x


def _train(x, y, cell, seed):
    """Fit one network; keep the epoch that does best on the most recent 15% of the samples."""
    torch.manual_seed(seed)
    n_val = max(int(0.15 * len(x)), 1)
    xt, yt = torch.tensor(x[:-n_val], dtype=torch.float32), torch.tensor(y[:-n_val], dtype=torch.float32)
    xv, yv = torch.tensor(x[-n_val:], dtype=torch.float32), torch.tensor(y[-n_val:], dtype=torch.float32)
    net = Net(cell, x.shape[2])
    opt = torch.optim.Adam(net.parameters(), lr=3e-3, weight_decay=1e-4)
    best, best_state = float("inf"), None
    for _ in range(EPOCHS):
        net.train()
        for i in torch.randperm(len(xt)).split(128):
            opt.zero_grad()
            nn.functional.mse_loss(net(xt[i]), yt[i]).backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            loss = nn.functional.mse_loss(net(xv), yv).item()
        if loss < best:
            best, best_state = loss, {k: v.clone() for k, v in net.state_dict().items()}
    net.load_state_dict(best_state)
    return net.eval()


def _predict(net, x):
    with torch.no_grad():
        return net(torch.tensor(x[None], dtype=torch.float32))[0].numpy()


# --- price task ------------------------------------------------------------------------------

def _price_fit(r, cell, seed):
    std = r.std()
    ends, x = _samples(r, lambda w, e: (w / std)[:, None])
    y = np.stack([[r[e:e + h].sum() / (std * np.sqrt(h)) for h in HORIZONS] for e in ends])
    return _train(x, y, cell, seed), std


def price_forecaster(cell, seed):
    """Price forecasts: last close times the exponential of the predicted cumulative return."""
    state = {"calls": 0}

    def forecast(hist, future_dates):
        r = np.diff(np.log(hist.to_numpy()))
        if state["calls"] % REFIT_EVERY == 0:
            state["net"], state["std"] = _price_fit(r, cell, seed)
        state["calls"] += 1
        out = _predict(state["net"], (r[-WINDOW:] / state["std"])[:, None])
        cum = out * state["std"] * np.sqrt(HORIZONS)
        path = np.interp(np.arange(1, LONGEST + 1), [0, *HORIZONS], [0.0, *cum])
        return hist.iloc[-1] * np.exp(path)
    return forecast


def run_price(name, cell, seeds=SEEDS):
    """One backtest per seed, then their average (the model's forecast)."""
    prices = common.load_prices(name)
    per_seed = [common.backtest(prices, price_forecaster(cell, s)) for s in seeds]
    mean = per_seed[0].copy()
    mean["forecast"] = np.mean([d["forecast"].to_numpy() for d in per_seed], axis=0)
    return per_seed, mean


# --- volatility task -------------------------------------------------------------------------

def _vol_builder(m, sd, extra):
    """Inputs for one 30-day window ending at position e: the returns and their log squares, and,
    when market information is used, the market's returns and the log fear index."""
    def build(w, e):
        cols = [w / sd, (np.log(w ** 2 + 1e-6) - m[0]) / m[1]]
        if extra is not None:
            cols += [extra[e - WINDOW:e, 0], extra[e - WINDOW:e, 1]]
        return np.stack(cols, axis=1)
    return build


def _standardise(raw, stats):
    return None if raw is None else np.nan_to_num((raw - stats[0]) / stats[1])


def _vol_fit(r, raw_extra, cell, seed):
    sd = r.std()
    logsq = np.log(r ** 2 + 1e-6)
    m = (logsq.mean(), logsq.std())
    stats = None if raw_extra is None else (np.nanmean(raw_extra, axis=0), np.nanstd(raw_extra, axis=0))
    ends, x = _samples(r, _vol_builder(m, sd, _standardise(raw_extra, stats)))
    y = np.stack([[np.log(np.mean(r[e:e + h] ** 2) + 1e-8) for h in HORIZONS] for e in ends])
    ym, ys = y.mean(axis=0), y.std(axis=0)
    net = _train(x, (y - ym) / ys, cell, seed)
    # exp(E[log v]) under-states E[v]; rescale so the forecasts are right on average in training
    pred = np.stack([_predict(net, xi) for xi in x]) * ys + ym
    smear = np.clip(np.exp(y).mean(axis=0) / np.exp(pred).mean(axis=0), 0.5, 2.0)
    return net, ym, ys, smear, m, sd, stats


def vol_forecaster(cell, seed, market_columns=None):
    """Variance path for the next 20 days whose averages over 1, 5 and 20 days are the predictions.

    `market_columns` (market return and log fear index, by date) gives the network market inputs.
    """
    state = {"calls": 0}

    def forecast(returns):
        r = returns.to_numpy()
        raw = None if market_columns is None else market_columns.reindex(returns.index).to_numpy()
        if state["calls"] % REFIT_EVERY == 0:
            state["fit"] = _vol_fit(r, raw, cell, seed)
        state["calls"] += 1
        net, ym, ys, smear, m, sd, stats = state["fit"]
        feats = _vol_builder(m, sd, _standardise(raw, stats))
        v1, v5, v20 = np.exp(_predict(net, feats(r[-WINDOW:], len(r))) * ys + ym) * smear
        return garch.path_from_averages(v1, v5, v20)
    return forecast


def run_vol(name, cell, seeds=SEEDS, market_info=False):
    prices = common.load_prices(name)
    columns = market.market_frame(name, prices.index)[["mkt_r0", "fear"]] if market_info else None
    per_seed = [garch.vol_backtest(prices, vol_forecaster(cell, s, columns)) for s in seeds]
    mean = per_seed[0].copy()
    mean["forecast"] = np.mean([d["forecast"].to_numpy() for d in per_seed], axis=0)
    return per_seed, mean


def seed_spread(per_seed, base_df, kind):
    """The error relative to the baseline, for each seed on its own."""
    rows = []
    for seed, df in zip(SEEDS, per_seed):
        if kind == "price":
            table = common.compare_to_naive({"Naive": base_df, "m": df}).query("model == 'm'")
            rows += [(seed, int(h), float(v)) for h, v in zip(table.horizon_days, table.rmse_vs_naive)]
        else:
            table = garch.compare_to_baseline({garch.BASELINE: base_df, "m": df}).query("model == 'm'")
            rows += [(seed, int(h), float(v)) for h, v in zip(table.horizon_days, table.rmse_vs_baseline)]
    return pd.DataFrame(rows, columns=["seed", "horizon_days", "ratio_vs_baseline"])


if __name__ == "__main__":
    import sys
    name = sys.argv[1] if len(sys.argv) > 1 else "PERSISTENT"
    naive_df = common.backtest(common.load_prices(name), common.naive)
    for cell in CELLS:
        per_seed, mean = run_price(name, cell)
        t = common.compare_to_naive({"Naive": naive_df, f"{cell} (average of seeds)": mean})
        print(f"\n{name} price, {cell}")
        print(t.round(3).to_string(index=False))
        print(seed_spread(per_seed, naive_df, "price").pivot(index="seed", columns="horizon_days",
                                                              values="ratio_vs_baseline").round(3))
