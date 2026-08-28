"""P0: build the joined Premier League panel and check for signal.

Alignment is the one the model will use: stats from season t predict the
Transfermarkt value recorded at the start of season t+1.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from scipy import stats as sps

from src.entity.resolve import resolve

ROOT = Path(__file__).resolve().parents[1]
MIN_MINUTES = 600


def main() -> pd.DataFrame:
    fb = pd.read_parquet(ROOT / "data/raw/fbref_standard.parquet")
    tm = pd.read_parquet(ROOT / "data/raw/tm_vals.parquet")

    fb = fb[(fb.Comp == "Premier League") & (fb.Season_End_Year.between(2018, 2023))]
    link = resolve(fb, tm)

    df = fb.merge(
        link[["season_end_year", "fb_player", "fb_squad", "tm_url", "tier"]],
        left_on=["Season_End_Year", "Player", "Squad"],
        right_on=["season_end_year", "fb_player", "fb_squad"],
        how="left",
    )

    # Label: TM value at the start of the FOLLOWING season.
    label = tm[["player_url", "season_start_year", "player_market_value_euro",
                "player_dob", "player_position", "contract_expiry"]].rename(
        columns={"player_market_value_euro": "value_eur"})
    df["label_season"] = df.Season_End_Year          # value at start of t+1
    df = df.merge(label, left_on=["tm_url", "label_season"],
                  right_on=["player_url", "season_start_year"], how="left")

    df = df[(df.Pos != "GK") & (df.Min_Playing >= MIN_MINUTES)].copy()
    df["npxg_xag_per90"] = df["npxG+xAG_Per"]
    # FBref stores Age as a string; TM dob gives a precise age at the snapshot.
    df["age"] = pd.to_numeric(df["Age"], errors="coerce")
    dob = pd.to_datetime(df.player_dob, errors="coerce")
    snap = pd.to_datetime(dict(year=df.label_season, month=7, day=1))
    df["age_exact"] = (snap - dob).dt.days / 365.25
    df["log_value"] = np.log10(df.value_eur)
    return df


if __name__ == "__main__":
    df = main()
    out = ROOT / "data/processed/p0_panel.parquet"
    df.to_parquet(out, index=False)

    n = len(df)
    lab = df.value_eur.notna().sum()
    print(f"panel rows (outfield, >={MIN_MINUTES} min): {n:,}")
    print(f"  with TM link      : {df.tm_url.notna().sum():,} ({df.tm_url.notna().mean()*100:.1f}%)")
    print(f"  with t+1 label    : {lab:,} ({lab/n*100:.1f}%)")
    print(f"  -> label loss is players who left the PL after season t\n")

    d = df.dropna(subset=["value_eur", "npxg_xag_per90"])
    for col, name in [("npxg_xag_per90", "npxG+xAG /90"), ("age_exact", "age"),
                      ("Min_Playing", "minutes")]:
        dd = d.dropna(subset=[col])
        r = sps.spearmanr(dd[col], dd.value_eur)
        print(f"  Spearman(log value, {name:<13}) = {r.statistic:+.3f}   p={r.pvalue:.1e}")

    print(f"\n  value range: EUR {d.value_eur.min()/1e6:.1f}m - {d.value_eur.max()/1e6:.0f}m"
          f"  median EUR {d.value_eur.median()/1e6:.1f}m")
    print("\nby season:")
    print(d.groupby("Season_End_Year").agg(
        n=("value_eur", "size"),
        median_val_m=("value_eur", lambda s: round(s.median()/1e6, 1)),
    ).to_string())
