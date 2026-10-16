"""End-to-end study: data -> shocks -> event study -> lead-lag -> trading test -> report.

    python -m pmes.pipeline --synthetic         # validate on markets with a known lag
    python -m pmes.pipeline --data-dir data     # real pairs listed in data/pairs.csv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import plots
from .analysis import (adjustment_profile, bootstrap_mean, cross_correlation, detect_shocks,
                       event_paths, fit_sensitivity, lead_lag_regressions)
from .config import Config
from .data import align, asset_hourly, load_pair, save_pair
from .strategy import summarize, trade_events


def fetch_real(cfg: Config, pairs: pd.DataFrame) -> None:
    """Download any pair whose data file is missing."""
    from .polymarket import price_history

    for _, p in pairs.iterrows():
        if (Path(cfg.data_dir) / f"{p['name']}.parquet").exists():
            continue
        print(f"  downloading {p['name']}: Polymarket {str(p['token_id'])[:12]}... vs {p['asset']}")
        prob = price_history(str(p["token_id"]), p["start"], p["end"])
        asset = asset_hourly(p["asset"], p["start"], p["end"])
        save_pair(align(prob, asset), cfg.data_dir, p["name"])


def run(cfg: Config) -> dict:
    report = Path(cfg.report_dir)
    report.mkdir(parents=True, exist_ok=True)
    pairs = pd.read_csv(cfg.pairs_path)

    all_paths, all_events, ccf, regs = {"prob": [], "asset": []}, [], [], {}
    oos_trades = {d: [] for d in cfg.delays}
    is_trades = {d: [] for d in cfg.delays}
    pair_rows = []

    for _, p in pairs.iterrows():
        df = load_pair(cfg.data_dir, p["name"])
        events = detect_shocks(df, cfg, sign=float(p["sign"]))
        n_cal = int(len(events) * cfg.split_frac)
        cal, test = events.iloc[:n_cal], events.iloc[n_cal:]
        beta = fit_sensitivity(df, cal, cfg.response_horizon)

        paths = event_paths(df, events, cfg.pre_hours, cfg.post_hours)
        for k in all_paths:
            all_paths[k].append(paths[k])
        all_events.append(events.assign(pair=p["name"]))
        ccf.append(cross_correlation(df, cfg.max_lag).rename(p["name"]))
        regs[p["name"]] = lead_lag_regressions(df)

        for d in cfg.delays:
            is_trades[d].append(trade_events(df, cal, beta, cfg, d).assign(pair=p["name"]))
            oos_trades[d].append(trade_events(df, test, beta, cfg, d).assign(pair=p["name"]))

        row = {"pair": p["name"], "bars": len(df), "events": len(events), "calibration_events": n_cal,
               "fitted_beta": round(beta, 2)}
        if "true_beta" in p:
            row["true_beta"] = p["true_beta"]
        pair_rows.append(row)
        print(f"  {p['name']}: {len(df)} bars, {len(events)} shocks, beta {beta:.2f}")

    prob_paths = pd.concat(all_paths["prob"])
    asset_paths = pd.concat(all_paths["asset"])
    prob_band = bootstrap_mean(prob_paths, cfg.n_bootstrap)
    asset_band = bootstrap_mean(asset_paths, cfg.n_bootstrap)
    profile = adjustment_profile(prob_band["mean"], cfg.response_horizon)
    ccf_avg = pd.concat(ccf, axis=1).mean(axis=1)

    def collect(trades):
        out = {}
        for d, lst in trades.items():
            tr = pd.concat(lst).sort_values("time") if lst else pd.DataFrame()
            out[d] = tr
        return out

    is_tr, oos_tr = collect(is_trades), collect(oos_trades)
    strat = {f"delay_{d}h": {"in_sample": summarize(is_tr[d]), "out_of_sample": summarize(oos_tr[d])}
             for d in cfg.delays}
    for d, tr in oos_tr.items():
        tr.to_csv(report / f"trades_oos_delay{d}.csv", index=False)
    pd.concat(all_events).to_csv(report / "events.csv", index=False)

    plots.event_path(prob_band, report / "event_study_prob.png",
                     "Polymarket probability after an asset shock (signed, all pairs)",
                     "Change from pre-shock level (pp)")
    plots.event_path(asset_band, report / "event_study_asset.png",
                     "Asset price after the same shocks (signed)", "Cumulative log return (%)")
    plots.lead_lag(ccf_avg, report / "lead_lag.png", "Lead-lag: who moves first? (average over pairs)")
    plots.pnl_by_delay(oos_tr, report / "trades_oos.png",
                       f"Out-of-sample net P&L by entry delay ({cfg.cost_prob * 100:.0f}pp costs)")

    results = {
        "pairs": pair_rows,
        "n_events": int(len(prob_paths)),
        "adjustment": profile,
        "cross_correlation": ccf_avg.round(4).to_dict(),
        "strategy": strat,
    }
    (report / "results.json").write_text(json.dumps(results, indent=2, default=float))
    write_summary(report, cfg, results, regs, prob_band)
    print(f"Done. See {report / 'summary.md'}")
    return results


def _f(x, d=2):
    return "–" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{d}f}"


def write_summary(report, cfg, res, regs, prob_band):
    a = res["adjustment"]
    L = ["# Results summary", "",
         f"Shocks: asset hourly return beyond {cfg.shock_z:g}× trailing volatility, "
         f"{cfg.cooldown_hours}h cooldown. Events: **{res['n_events']}**.", "",
         "## Pairs", "", "| Pair | Bars | Shocks | Calibration events | Fitted β (logit / log-return) |"
         + (" True β |" if "true_beta" in res["pairs"][0] else ""),
         "|---|---|---|---|---|" + ("---|" if "true_beta" in res["pairs"][0] else "")]
    for p in res["pairs"]:
        L.append(f"| {p['pair']} | {p['bars']} | {p['events']} | {p['calibration_events']} | {p['fitted_beta']} |"
                 + (f" {p['true_beta']} |" if "true_beta" in p else ""))
    L += ["", "## How fast does Polymarket adjust?", "",
          f"- Signed probability move on the shock bar: **{a['move_at_shock_bar'] * 100:.2f} pp**",
          f"- Move after {cfg.response_horizon}h: **{a['move_at_horizon'] * 100:.2f} pp**",
          f"- Share of the {cfg.response_horizon}h move already done on the shock bar: **{a['share_on_shock_bar']:.0%}**",
          f"- Bars until half the move is done: **{a['half_life_bars']}**",
          f"- Post-shock drift (the part a trader could still capture): **{a['post_shock_drift'] * 100:.2f} pp**", "",
          "Mean signed path (pp): " + ", ".join(f"{k:+d}h {v * 100:.2f}" for k, v in prob_band["mean"].loc[0:6].items()),
          "", "## Lead-lag regressions (HC1 t-stats, first pair shown)", ""]
    first = next(iter(regs))
    for name, tbl in regs[first].items():
        L += [f"**{name}** ({first})", "", "| Term | Coef | t |", "|---|---|---|"]
        L += [f"| {i} | {r.coef:.4f} | {r.t:.2f} |" for i, r in tbl.iterrows()]
        L.append("")
    L += ["Cross-correlation corr(Δlogit_t, r_t−k), averaged over pairs (k > 0: asset leads): "
          + ", ".join(f"k={k}: {v:+.3f}" for k, v in res["cross_correlation"].items()), "",
          "## Trading test (buy the lagging contract)", "",
          f"Enter when |implied − market| > {cfg.entry_threshold * 100:.0f}pp, hold {cfg.hold_hours}h, "
          f"{cfg.cost_prob * 100:.0f}pp round-trip cost. Sensitivity fitted on the first "
          f"{cfg.split_frac:.0%} of each pair's events.", "",
          "| Delay | Sample | Trades | Hit rate | Mean P&L (pp) | t-stat | Mean return |",
          "|---|---|---|---|---|---|---|"]
    for key, v in res["strategy"].items():
        for sample, s in v.items():
            L.append(f"| {key.replace('delay_', '')} | {sample.replace('_', '-')} | {s['n_trades']} | "
                     f"{_f(s['hit_rate'] * 100 if s['n_trades'] else np.nan, 0)}% | {_f(s['mean_pnl_pp'])} | "
                     f"{_f(s['t_stat'])} | {_f(s['mean_return'] * 100 if s['n_trades'] else np.nan, 1)}% |")
    L += ["", "![Event study](event_study_prob.png)", "", "![Asset](event_study_asset.png)", "",
          "![Lead-lag](lead_lag.png)", "", "![Trades](trades_oos.png)", ""]
    (report / "summary.md").write_text("\n".join(L))


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", default=None)
    p.add_argument("--report-dir", default=None)
    p.add_argument("--synthetic", action="store_true")
    p.add_argument("--cost-pp", type=float, default=None, help="Round-trip cost in probability points")
    args = p.parse_args(argv)

    cfg = Config()
    if args.synthetic:
        cfg.data_dir = Path(args.data_dir or "data/synthetic")
        cfg.report_dir = Path(args.report_dir or "reports/synthetic")
        if not cfg.pairs_path.exists():
            from .synthetic import generate
            generate(cfg.data_dir)
    else:
        cfg.data_dir = Path(args.data_dir or "data")
        cfg.report_dir = Path(args.report_dir or "reports")
        fetch_real(cfg, pd.read_csv(cfg.pairs_path))
    if args.cost_pp is not None:
        cfg.cost_prob = args.cost_pp / 100
    run(cfg)


if __name__ == "__main__":
    main()
