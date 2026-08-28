"""Build labels from real Transfermarkt value histories.

The mirror carried one valuation per season and its 2022 snapshot was a stale
copy. The scraped histories are dated, so labels can be defined precisely:

    season t  = Season_End_Year Y, running roughly Aug (Y-1) to May Y
    label     = first valuation dated on or after 1 June Y   (post-season revaluation)
    prior     = last valuation dated strictly before 1 Aug (Y-1)  (pre-season)

Nothing dated inside or after the label window can reach the feature matrix.
"""
from __future__ import annotations

import pandas as pd

from src.ingest.transfermarkt import load_history, player_id

LABEL_MONTH, LABEL_DAY = 6, 1
PRIOR_MONTH, PRIOR_DAY = 8, 1


def _pick(series: pd.DataFrame, cutoff: pd.Timestamp, after: bool):
    if after:
        hit = series[series.value_date >= cutoff]
        return hit.iloc[0] if len(hit) else None
    hit = series[series.value_date < cutoff]
    return hit.iloc[-1] if len(hit) else None


def build_labels(seasons: list[int]) -> pd.DataFrame:
    """One row per (tm_player_id, Season_End_Year) with label and prior value."""
    hist = load_history()
    if hist.empty:
        return pd.DataFrame()

    rows = []
    for pid, g in hist.groupby("tm_player_id", sort=False):
        g = g.sort_values("value_date")
        for y in seasons:
            lab = _pick(g, pd.Timestamp(y, LABEL_MONTH, LABEL_DAY), after=True)
            pri = _pick(g, pd.Timestamp(y - 1, PRIOR_MONTH, PRIOR_DAY), after=False)
            if lab is None:
                continue
            rows.append({
                "tm_player_id": pid,
                "Season_End_Year": y,
                "value_eur_real": float(lab.value_eur),
                "label_date": lab.value_date,
                "prior_value_eur_real": float(pri.value_eur) if pri is not None else None,
                "prior_date": pri.value_date if pri is not None else pd.NaT,
            })
    return pd.DataFrame(rows)


def attach_real_labels(df: pd.DataFrame, seasons: list[int]) -> pd.DataFrame:
    """Replace mirror-derived labels with the scraped, dated ones where available."""
    labels = build_labels(seasons)
    if labels.empty:
        df["label_source"] = "mirror"
        return df

    df = df.copy()
    df["tm_player_id"] = df.tm_url.map(lambda u: player_id(u) if isinstance(u, str) else None)
    df = df.merge(labels, on=["tm_player_id", "Season_End_Year"], how="left")

    have = df.value_eur_real.notna()
    df["label_source"] = pd.Series(["mirror"] * len(df), index=df.index).mask(have, "scraped")
    df["value_eur"] = df.value_eur_real.where(have, df.value_eur)
    df["prior_value_eur"] = df.prior_value_eur_real.where(
        df.prior_value_eur_real.notna(), df.prior_value_eur)
    return df.drop(columns=["value_eur_real", "prior_value_eur_real"])
