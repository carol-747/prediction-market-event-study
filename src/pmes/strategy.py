"""Trading test: buy the lagging Polymarket contract after an asset shock.

For each shock at bar i, wait `delay` bars (latency), then compare the market
probability with the implied probability. If the gap exceeds `entry_threshold`,
buy Yes (gap > 0) or No (gap < 0) and sell after `hold_hours` bars.

P&L per share is in probability points (a share pays $1), net of a round-trip
cost covering the bid-ask spread and fees. Return on capital divides by the
price paid for the share.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .analysis import implied_prob
from .config import Config


def trade_events(df: pd.DataFrame, events: pd.DataFrame, beta: float, cfg: Config, delay: int) -> pd.DataFrame:
    prob = df["prob"].to_numpy()
    rows = []
    for _, e in events.iterrows():
        i = int(e["i"])
        entry, exit_ = i + delay, i + delay + cfg.hold_hours
        if i < 1 or exit_ >= len(df):
            continue
        target = implied_prob(df, i, e["r"], beta)
        p_in, p_out = prob[entry], prob[exit_]
        gap = target - p_in
        if abs(gap) < cfg.entry_threshold:
            continue
        side = 1.0 if gap > 0 else -1.0
        pnl = side * (p_out - p_in) - cfg.cost_prob
        price_paid = p_in if side > 0 else 1 - p_in
        rows.append({"time": e["time"], "side": "YES" if side > 0 else "NO", "p_entry": p_in,
                     "p_exit": p_out, "implied": target, "gap": gap, "pnl": pnl,
                     "return": pnl / price_paid})
    return pd.DataFrame(rows, columns=["time", "side", "p_entry", "p_exit", "implied", "gap", "pnl", "return"])


def summarize(trades: pd.DataFrame) -> dict:
    n = len(trades)
    if n == 0:
        return {"n_trades": 0, "hit_rate": np.nan, "mean_pnl_pp": np.nan, "t_stat": np.nan,
                "mean_return": np.nan, "total_pnl_pp": 0.0}
    pnl = trades["pnl"]
    t = pnl.mean() / pnl.std() * np.sqrt(n) if n > 1 and pnl.std() > 0 else np.nan
    return {"n_trades": n, "hit_rate": float((pnl > 0).mean()), "mean_pnl_pp": float(pnl.mean() * 100),
            "t_stat": float(t), "mean_return": float(trades["return"].mean()),
            "total_pnl_pp": float(pnl.sum() * 100)}
