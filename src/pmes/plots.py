"""Static report charts (PNG)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

SURFACE, INK, INK_2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]


def _style(ax, title, ylabel, xlabel=None):
    ax.set_facecolor(SURFACE)
    ax.figure.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", color=INK, fontsize=12, fontweight="bold", pad=12)
    ax.set_ylabel(ylabel, color=INK_2, fontsize=9)
    if xlabel:
        ax.set_xlabel(xlabel, color=INK_2, fontsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=8, length=0)


def _save(fig, path):
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", pad_inches=0.2, dpi=150)
    plt.close(fig)


def event_path(band: pd.DataFrame, path: Path, title: str, ylabel: str, scale: float = 100):
    fig, ax = plt.subplots(figsize=(8, 4))
    x = band.index.to_numpy()
    ax.fill_between(x, band["lo"] * scale, band["hi"] * scale, color=SERIES[0], alpha=0.15, linewidth=0,
                    label="95% bootstrap CI")
    ax.plot(x, band["mean"] * scale, color=SERIES[0], linewidth=2, marker="o", markersize=3, label="Mean")
    ax.axvline(0, color=MUTED, linewidth=0.8, linestyle="--")
    ax.axhline(0, color=MUTED, linewidth=0.8)
    ax.annotate("shock bar", (0, ax.get_ylim()[1]), xytext=(4, -10), textcoords="offset points",
                color=INK_2, fontsize=8)
    _style(ax, title, ylabel, "Hours relative to the asset shock")
    ax.legend(frameon=False, fontsize=8, labelcolor=INK_2, loc="upper left")
    _save(fig, path)


def lead_lag(corr: pd.Series, path: Path, title: str):
    fig, ax = plt.subplots(figsize=(8, 3.8))
    colors = [SERIES[0] if k > 0 else (SERIES[1] if k < 0 else MUTED) for k in corr.index]
    ax.bar(corr.index, corr.values, color=colors, width=0.7, edgecolor=SURFACE, linewidth=2)
    ax.axhline(0, color=MUTED, linewidth=0.8)
    _style(ax, title, "corr(Δ logit prob_t, asset return_t−k)", "Lag k (hours)")
    ymax = ax.get_ylim()[1]
    ax.annotate("asset leads →", (corr.index.max(), ymax), ha="right", va="top", color=SERIES[0], fontsize=9)
    ax.annotate("← prob leads", (corr.index.min(), ymax), ha="left", va="top", color=SERIES[1], fontsize=9)
    _save(fig, path)


def pnl_by_delay(trades_by_delay: dict[int, pd.DataFrame], path: Path, title: str):
    """Mean net P&L per trade for each entry delay, with the trade count under each bar."""
    labels, means, counts = [], [], []
    for delay, tr in trades_by_delay.items():
        labels.append(f"{delay}h")
        means.append(tr["pnl"].mean() * 100 if len(tr) else 0.0)
        counts.append(len(tr))
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    bars = ax.bar(labels, means, color=SERIES[0], width=0.55, edgecolor=SURFACE, linewidth=2)
    for b, m, n in zip(bars, means, counts):
        ax.annotate(f"{m:+.1f} pp\n{n} trades", (b.get_x() + b.get_width() / 2, m),
                    xytext=(0, 4 if m >= 0 else -4), textcoords="offset points", ha="center",
                    va="bottom" if m >= 0 else "top", color=INK_2, fontsize=8)
    lo, hi = ax.get_ylim()
    pad = (hi - lo) * 0.25
    ax.set_ylim(min(lo, min(means) - pad, 0), max(hi, max(means) + pad))
    ax.axhline(0, color=MUTED, linewidth=0.8)
    _style(ax, title, "Mean net P&L per trade (pp)", "Delay between the asset shock and the trade")
    _save(fig, path)
