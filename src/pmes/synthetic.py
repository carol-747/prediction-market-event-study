"""Synthetic markets with a KNOWN lag, to test the analysis end to end.

The asset follows noisy hourly returns with occasional jumps. Each Polymarket
probability responds to asset returns spread over the current and next three
bars (25% / 35% / 25% / 15%), plus its own news jumps and noise. So the asset
leads by construction, and roughly three quarters of the response arrives
AFTER the shock bar. The analysis should recover both facts.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .data import align, inv_logit, save_pair

LAG_WEIGHTS = np.array([0.25, 0.35, 0.25, 0.15])

PAIRS = [
    # name, true sensitivity (logit per log return), start prob
    ("escalation_vs_oil", 15.0, 0.40),
    ("ceasefire_vs_oil", -12.0, 0.55),
    ("rate_cut_vs_bonds", 10.0, 0.35),
]


def simulate_pair(beta: float, p0: float, n: int = 4000, seed: int = 0,
                  lag_weights: np.ndarray = LAG_WEIGHTS) -> tuple[pd.Series, pd.Series]:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2025-01-06", periods=n, freq="h", tz="UTC")
    r = rng.standard_t(5, n) * 0.003
    jumps = rng.random(n) < 0.006
    r[jumps] += rng.normal(0, 0.025, jumps.sum())
    asset = 70 * np.exp(np.cumsum(r))

    driven = np.convolve(r, lag_weights)[:n] * beta
    own_news = (rng.random(n) < 0.003) * rng.normal(0, 0.4, n)
    x0 = np.log(p0 / (1 - p0))
    x = np.empty(n)
    x_prev = x0
    for t in range(n):
        x_prev = x_prev + driven[t] + own_news[t] + rng.normal(0, 0.03) - 0.003 * (x_prev - x0)
        x[t] = x_prev
    prob = np.round(inv_logit(x), 3)  # Polymarket quotes in 0.1 cent ticks
    return pd.Series(prob, index=idx, name="prob"), pd.Series(asset, index=idx, name="asset")


def generate(data_dir: Path | str, n: int = 8000, seed: int = 11) -> pd.DataFrame:
    data_dir = Path(data_dir)
    rows = []
    for k, (name, beta, p0) in enumerate(PAIRS):
        prob, asset = simulate_pair(beta, p0, n=n, seed=seed + k)
        save_pair(align(prob, asset), data_dir, name)
        rows.append({"name": name, "token_id": "synthetic", "asset": "synthetic",
                     "sign": int(np.sign(beta)), "true_beta": beta})
    pairs = pd.DataFrame(rows)
    pairs.to_csv(data_dir / "pairs.csv", index=False)
    return pairs
