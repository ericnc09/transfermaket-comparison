"""Fetch real Transfermarkt value histories for every player in the panel."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src.ingest.transfermarkt import fetch_many, player_id

ROOT = Path(__file__).resolve().parents[1]

if __name__ == "__main__":
    panel = pd.read_parquet(ROOT / "data/processed/panel_full.parquet")
    urls = panel.loc[panel.eligible, "tm_url"].dropna().unique()
    pids = sorted({p for p in (player_id(u) for u in urls) if p})
    print(f"{len(pids):,} unique players to cover", flush=True)
    fetch_many(pids)
