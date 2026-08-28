"""Ingest for openly-published FBref and Transfermarkt mirrors.

FBref's own site sits behind a Cloudflare JS challenge, so we read the
worldfootballR_data mirror instead. It publishes the same Opta-derived
season tables as .rds, alongside Transfermarkt squad valuations.
"""
from __future__ import annotations

import os
import tempfile
import urllib.request
from pathlib import Path

import pandas as pd
import pyreadr

RAW = Path(__file__).resolve().parents[2] / "data" / "raw"
BASE = "https://github.com/JaseZiv/worldfootballR_data/raw/master/data"

FBREF_TABLES = [
    "standard", "shooting", "passing", "passing_types",
    "defense", "possession", "gca", "misc", "playing_time",
]

SOURCES: dict[str, str] = {
    **{f"fbref_{t}": f"{BASE}/fb_big5_advanced_season_stats/big5_player_{t}.rds"
       for t in FBREF_TABLES},
    "tm_vals": f"{BASE}/tm_player_vals/big5_player_vals.rds",
}


def _fetch_rds(url: str) -> pd.DataFrame:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=180) as resp:
        raw = resp.read()
    tmp = tempfile.NamedTemporaryFile(suffix=".rds", delete=False)
    try:
        tmp.write(raw)
        tmp.close()
        return list(pyreadr.read_r(tmp.name).values())[0]
    finally:
        os.unlink(tmp.name)


def load(name: str, refresh: bool = False) -> pd.DataFrame:
    """Return a source table, downloading to a local parquet cache on first use."""
    RAW.mkdir(parents=True, exist_ok=True)
    cache = RAW / f"{name}.parquet"
    if cache.exists() and not refresh:
        return pd.read_parquet(cache)
    df = _fetch_rds(SOURCES[name])
    df.to_parquet(cache, index=False)
    return df


def load_all(refresh: bool = False) -> dict[str, pd.DataFrame]:
    return {name: load(name, refresh) for name in SOURCES}
