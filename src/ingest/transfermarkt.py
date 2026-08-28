"""Fetch real Transfermarkt market-value histories.

The worldfootballR mirror carries one valuation per season, and its 2022 snapshot
is a 99.5% copy of 2021 - unusable as a label. Transfermarkt's own chart endpoint
returns the full dated series per player, which repairs that season and restores
the roughly three-updates-a-year cadence the design originally called for.

Cached on disk, so a re-run costs nothing and an interrupted crawl resumes.
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "data" / "raw" / "tm_value_history"
ENDPOINT = "https://www.transfermarkt.com/ceapi/marketValueDevelopment/graph/{pid}"
HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                   "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"),
    "Accept": "application/json",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.transfermarkt.com/",
}
DELAY = 1.2          # seconds between requests - deliberately polite
_ID = re.compile(r"/spieler/(\d+)")


def player_id(tm_url: str) -> str | None:
    m = _ID.search(tm_url or "")
    return m.group(1) if m else None


def fetch_one(pid: str, session: requests.Session) -> list[dict] | None:
    """Return the raw value series for one player, or None if unavailable."""
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / f"{pid}.json"
    if path.exists():
        try:
            return json.loads(path.read_text()).get("list")
        except json.JSONDecodeError:
            path.unlink()

    for attempt in range(4):
        try:
            r = session.get(ENDPOINT.format(pid=pid), headers=HEADERS, timeout=25)
        except requests.RequestException:
            time.sleep(2 ** attempt)
            continue
        if r.status_code == 200:
            path.write_text(r.text)
            time.sleep(DELAY)
            try:
                return r.json().get("list")
            except json.JSONDecodeError:
                return None
        if r.status_code in (429, 503):
            time.sleep(int(r.headers.get("Retry-After", 15)) * (attempt + 1))
            continue
        time.sleep(DELAY)
        return None
    return None


def fetch_many(pids: list[str], log_every: int = 100) -> int:
    CACHE.mkdir(parents=True, exist_ok=True)
    todo = [p for p in pids if not (CACHE / f"{p}.json").exists()]
    print(f"{len(pids):,} players, {len(pids) - len(todo):,} cached, {len(todo):,} to fetch",
          flush=True)
    ok = 0
    with requests.Session() as s:
        for i, pid in enumerate(todo, 1):
            if fetch_one(pid, s) is not None:
                ok += 1
            if i % log_every == 0:
                print(f"  {i:,}/{len(todo):,}  ok={ok:,}", flush=True)
    return ok


def load_history() -> pd.DataFrame:
    """Flatten every cached series into (player_id, date, value_eur, club, age)."""
    rows = []
    for f in sorted(CACHE.glob("*.json")):
        try:
            series = json.loads(f.read_text()).get("list") or []
        except json.JSONDecodeError:
            continue
        for e in series:
            if e.get("y") is None:
                continue
            rows.append({
                "tm_player_id": f.stem,
                # `x` is epoch milliseconds - unambiguous, unlike the DD/MM/YYYY
                # display string, which is locale-dependent.
                "epoch_ms": e.get("x"),
                "value_eur": float(e["y"]),
                "club": e.get("verein"),
                "age_at_value": pd.to_numeric(e.get("age"), errors="coerce"),
            })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["value_date"] = pd.to_datetime(df.epoch_ms, unit="ms", errors="coerce")
    return (df.dropna(subset=["value_date"])
              .drop(columns="epoch_ms")
              .sort_values(["tm_player_id", "value_date"]))
