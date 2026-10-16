"""All research settings in one place."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class Config:
    data_dir: Path = Path("data")
    report_dir: Path = Path("reports")

    # Shock detection on the traditional asset (hourly bars)
    vol_window: int = 120          # hours of trailing returns for the volatility estimate
    shock_z: float = 3.0           # |return| / trailing vol threshold
    min_abs_return: float = 0.005  # ignore tiny moves even if vol is very low
    cooldown_hours: int = 12       # one event per cluster of shocks
    prob_bounds: tuple[float, float] = (0.03, 0.97)  # need room for the probability to move

    # Event study
    pre_hours: int = 6
    post_hours: int = 24
    response_horizon: int = 6      # bars over which the "full" response is measured (calibration)
    max_lag: int = 6               # lead-lag analysis, hours
    n_bootstrap: int = 2000

    # Trading rule (trade the Polymarket contract)
    split_frac: float = 0.5        # first half of events: calibration; second half: out-of-sample
    entry_threshold: float = 0.05  # |implied prob - market prob| needed to trade
    hold_hours: int = 6
    delays: tuple[int, ...] = (0, 1, 2)  # hours between seeing the shock and trading
    cost_prob: float = 0.02        # round-trip cost in probability points (spread + fees)

    @property
    def pairs_path(self) -> Path:
        return Path(self.data_dir) / "pairs.csv"
