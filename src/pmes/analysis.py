"""Shock detection on the asset, event-study paths, lead-lag tests and sensitivity.

Hypothesis: informed traders move the traditional asset (e.g. crude oil) first;
Polymarket's probability adjusts with a lag. If true:
  * after an asset shock, the Polymarket probability keeps drifting in the
    implied direction for several hours (event study);
  * past asset returns predict today's probability change, but not the other
    way round (lead-lag).

Offsets are counted in asset bars (one bar = one hour while the asset trades).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import Config
from .data import inv_logit


# ------------------------------------------------------------------ shocks

def detect_shocks(df: pd.DataFrame, cfg: Config, sign: float = 1.0) -> pd.DataFrame:
    """Asset return shocks: |r| > shock_z * trailing vol (vol uses data before the bar only).

    Returns one row per event with its bar position `i`, the shock return and
    `direction` = the sign the probability is expected to move (+1 up, -1 down).
    """
    r = df["r"]
    vol = r.rolling(cfg.vol_window, min_periods=cfg.vol_window // 2).std().shift(1)
    z = r / vol
    prob_before = df["prob"].shift(1)
    lo, hi = cfg.prob_bounds
    cand = (z.abs() > cfg.shock_z) & (r.abs() > cfg.min_abs_return) & prob_before.between(lo, hi)

    events, last_time = [], None
    times = df.index
    for i in np.flatnonzero(cand.to_numpy()):
        t = times[i]
        if last_time is not None and (t - last_time) < pd.Timedelta(hours=cfg.cooldown_hours):
            continue
        last_time = t
        events.append({"time": t, "i": int(i), "r": r.iloc[i], "z": z.iloc[i],
                       "prob_before": prob_before.iloc[i], "direction": float(np.sign(r.iloc[i]) * sign)})
    return pd.DataFrame(events, columns=["time", "i", "r", "z", "prob_before", "direction"])


# ------------------------------------------------------------------ event study

def event_paths(df: pd.DataFrame, events: pd.DataFrame, pre: int, post: int) -> dict[str, pd.DataFrame]:
    """Signed paths around each event, measured from the bar BEFORE the shock (offset -1).

    prob: direction * (prob[t+k] - prob[t-1])           in probability points
    asset: sign(r_shock) * (log asset[t+k] - log asset[t-1])
    """
    offsets = np.arange(-pre, post + 1)
    prob = df["prob"].to_numpy()
    la = np.log(df["asset"].to_numpy())
    p_rows, a_rows, keep = [], [], []
    for _, e in events.iterrows():
        i = int(e["i"])
        if i - pre - 1 < 0 or i + post >= len(df):
            continue
        p_rows.append(e["direction"] * (prob[i + offsets] - prob[i - 1]))
        a_rows.append(np.sign(e["r"]) * (la[i + offsets] - la[i - 1]))
        keep.append(e["time"])
    cols = pd.Index(offsets, name="offset")
    return {"prob": pd.DataFrame(p_rows, index=keep, columns=cols),
            "asset": pd.DataFrame(a_rows, index=keep, columns=cols)}


def bootstrap_mean(paths: pd.DataFrame, n_boot: int = 2000, seed: int = 0) -> pd.DataFrame:
    """Mean path with a 95% bootstrap confidence band (resampling events)."""
    x = paths.to_numpy()
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(n_boot, len(x)))
    boots = x[idx].mean(axis=1)
    return pd.DataFrame({"mean": x.mean(axis=0), "lo": np.percentile(boots, 2.5, axis=0),
                         "hi": np.percentile(boots, 97.5, axis=0)}, index=paths.columns)


def adjustment_profile(mean_path: pd.Series, horizon: int) -> dict:
    """How much of the eventual (offset=horizon) move has happened by each offset."""
    total = mean_path.loc[horizon]
    post = mean_path.loc[0:horizon]
    frac = post / total if total != 0 else post * np.nan
    reached = frac[frac >= 0.5]
    return {
        "move_at_shock_bar": float(mean_path.loc[0]),
        "move_at_horizon": float(total),
        "share_on_shock_bar": float(frac.loc[0]),
        "half_life_bars": int(reached.index[0]) if len(reached) else None,
        "post_shock_drift": float(total - mean_path.loc[0]),
    }


# ------------------------------------------------------------------ lead-lag

def cross_correlation(df: pd.DataFrame, max_lag: int) -> pd.Series:
    """corr(dx_t, r_{t-k}). Positive k: the asset LEADS the probability by k bars."""
    out = {k: df["dx"].corr(df["r"].shift(k)) for k in range(-max_lag, max_lag + 1)}
    return pd.Series(out, name="corr").rename_axis("lag")


def _ols(y: np.ndarray, X: np.ndarray, names: list[str]) -> pd.DataFrame:
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    n, k = X.shape
    # Heteroskedasticity-robust (HC1) standard errors: event hours are far noisier.
    xtx_inv = np.linalg.inv(X.T @ X)
    meat = (X * resid[:, None] ** 2).T @ X
    cov = xtx_inv @ meat @ xtx_inv * n / (n - k)
    se = np.sqrt(np.diag(cov))
    return pd.DataFrame({"coef": beta, "se": se, "t": beta / se}, index=names)


def lead_lag_regressions(df: pd.DataFrame, lags: int = 3) -> dict[str, pd.DataFrame]:
    """Both directions, each controlling for the target's own lag.

    prob_on_asset: dx_t = a + sum_k b_k r_{t-k} + c dx_{t-1}
    asset_on_prob: r_t  = a + sum_k b_k dx_{t-k} + c r_{t-1}
    """
    out = {}
    for name, (y, x) in {"prob_on_asset": ("dx", "r"), "asset_on_prob": ("r", "dx")}.items():
        cols = {f"{x}_lag{k}": df[x].shift(k) for k in range(0, lags + 1)}
        cols[f"{y}_lag1"] = df[y].shift(1)
        data = pd.DataFrame({"y": df[y], **cols}).dropna()
        X = np.column_stack([np.ones(len(data))] + [data[c] for c in cols])
        out[name] = _ols(data["y"].to_numpy(), X, ["const"] + list(cols))
    return out


# ------------------------------------------------------------------ sensitivity

def fit_sensitivity(df: pd.DataFrame, events: pd.DataFrame, horizon: int) -> float:
    """Logit move per unit asset log return, from the FULL response over `horizon` bars.

    beta = sum(y * r) / sum(r^2), with y = logit(prob[t+h]) - logit(prob[t-1]).
    Fitted on calibration events only; used to compute the implied probability.
    """
    x = df["x"].to_numpy()
    ys, rs = [], []
    for _, e in events.iterrows():
        i = int(e["i"])
        if i - 1 < 0 or i + horizon >= len(df):
            continue
        ys.append(x[i + horizon] - x[i - 1])
        rs.append(e["r"])
    ys, rs = np.array(ys), np.array(rs)
    return float((ys * rs).sum() / (rs ** 2).sum()) if len(rs) else float("nan")


def implied_prob(df: pd.DataFrame, i: int, r: float, beta: float) -> float:
    """Where the probability 'should' go after the shock, starting from the pre-shock level."""
    return float(inv_logit(df["x"].iloc[i - 1] + beta * r))
