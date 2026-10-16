"""Polymarket public data: market search (Gamma API) and price history (CLOB API).

No API key is needed.

    python -m pmes.polymarket search "strait of hormuz"
"""

from __future__ import annotations

import argparse
import json
import time

import pandas as pd
import requests

GAMMA_SEARCH = "https://gamma-api.polymarket.com/public-search"
CLOB_HISTORY = "https://clob.polymarket.com/prices-history"


def search(query: str, session: requests.Session | None = None) -> pd.DataFrame:
    """Markets matching `query`, with the Yes-token id needed for price history."""
    s = session or requests.Session()
    resp = s.get(GAMMA_SEARCH, params={"q": query}, timeout=30)
    resp.raise_for_status()
    rows = []
    for event in resp.json().get("events", []):
        for m in event.get("markets", []):
            try:
                tokens = json.loads(m.get("clobTokenIds") or "[]")
                outcomes = json.loads(m.get("outcomes") or "[]")
            except (TypeError, json.JSONDecodeError):
                continue
            if not tokens:
                continue
            rows.append({
                "question": m.get("question"),
                "yes_outcome": outcomes[0] if outcomes else None,
                "yes_token_id": tokens[0],
                "start": m.get("startDate"),
                "end": m.get("endDate"),
                "closed": m.get("closed"),
                "volume": float(m.get("volume") or 0),
            })
    return pd.DataFrame(rows).sort_values("volume", ascending=False, ignore_index=True)


def price_history(token_id: str, start: str, end: str, fidelity_minutes: int = 60,
                  chunk_days: int = 14, session: requests.Session | None = None) -> pd.Series:
    """Yes-price history as a UTC-indexed Series, fetched in chunks."""
    s = session or requests.Session()
    t0, t1 = pd.Timestamp(start, tz="UTC"), pd.Timestamp(end, tz="UTC")
    points = []
    cur = t0
    while cur < t1:
        nxt = min(cur + pd.Timedelta(days=chunk_days), t1)
        params = {"market": token_id, "startTs": int(cur.timestamp()), "endTs": int(nxt.timestamp()),
                  "fidelity": fidelity_minutes}
        resp = s.get(CLOB_HISTORY, params=params, timeout=30)
        if resp.status_code == 429:
            time.sleep(float(resp.json().get("retry_after_seconds", 5)))
            continue
        resp.raise_for_status()
        points += resp.json().get("history", [])
        cur = nxt
    if not points:
        return pd.Series(dtype=float, name="prob")
    df = pd.DataFrame(points).drop_duplicates("t").sort_values("t")
    return pd.Series(df["p"].to_numpy(float), index=pd.to_datetime(df["t"], unit="s", utc=True), name="prob")


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="Search Polymarket markets")
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("search")
    s.add_argument("query")
    s.add_argument("--top", type=int, default=15)
    args = p.parse_args(argv)
    with pd.option_context("display.max_colwidth", 70, "display.width", 200):
        print(search(args.query).head(args.top).to_string())


if __name__ == "__main__":
    main()
