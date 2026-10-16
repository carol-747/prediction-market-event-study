"""Asset prices and alignment of Polymarket probabilities to the asset's hourly bars."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def asset_hourly(ticker: str, start: str, end: str) -> pd.Series:
    """Hourly closes from Yahoo Finance (Yahoo keeps roughly the last 730 days of hourly data)."""
    import yfinance as yf

    df = yf.download(ticker, start=start, end=end, interval="60m", auto_adjust=True, progress=False)
    close = df["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    close.index = pd.to_datetime(close.index, utc=True)
    return close.rename("asset").dropna()


def logit(p: pd.Series | np.ndarray, eps: float = 1e-3):
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


def inv_logit(x):
    return 1 / (1 + np.exp(-x))


def align(prob: pd.Series, asset: pd.Series) -> pd.DataFrame:
    """One row per asset bar: asset close and the LAST Polymarket price at or before that bar.

    Polymarket trades 24/7, the asset does not; aligning on the asset's bars means
    every row compares two prices observed at the same moment (no look-ahead).
    """
    prob = prob.sort_index()
    asset = asset.sort_index()
    p_at_bar = prob.reindex(prob.index.union(asset.index)).ffill().reindex(asset.index)
    df = pd.DataFrame({"asset": asset, "prob": p_at_bar}).dropna()
    df["r"] = np.log(df["asset"]).diff()
    df["x"] = logit(df["prob"])
    df["dx"] = df["x"].diff()
    return df.dropna()


def load_pair(data_dir: Path, name: str) -> pd.DataFrame:
    return pd.read_parquet(Path(data_dir) / f"{name}.parquet")


def save_pair(df: pd.DataFrame, data_dir: Path, name: str) -> None:
    Path(data_dir).mkdir(parents=True, exist_ok=True)
    df.to_parquet(Path(data_dir) / f"{name}.parquet")
