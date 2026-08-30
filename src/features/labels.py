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


def forward_values(pairs: pd.DataFrame, horizon_days: int = 365,
                   tolerance_days: int = 120) -> pd.DataFrame:
    """Transfermarkt value one year after each label date.

    `pairs` needs tm_player_id and label_date. Returns the first valuation dated
    at least `horizon_days` later, provided one exists within `tolerance_days`
    of that target - otherwise the row gets no forward value rather than a
    stale one.
    """
    hist = load_history()
    if hist.empty:
        return pd.DataFrame()

    by_player = {pid: g.sort_values("value_date")
                 for pid, g in hist.groupby("tm_player_id", sort=False)}
    out = []
    for r in pairs.itertuples():
        g = by_player.get(r.tm_player_id)
        if g is None or pd.isna(r.label_date):
            continue
        target = r.label_date + pd.Timedelta(days=horizon_days)
        later = g[g.value_date >= target]
        if later.empty:
            continue
        hit = later.iloc[0]
        if (hit.value_date - target).days > tolerance_days:
            continue
        # A zero valuation means Transfermarkt stopped pricing the player, not
        # that he became worthless. Left in, log1p(0) turns a EUR10m player into
        # a -16 log-point "return" and a handful of rows dominate the fit.
        if hit.value_eur <= 0:
            continue
        out.append({"tm_player_id": r.tm_player_id, "label_date": r.label_date,
                    "fwd_value_eur": float(hit.value_eur),
                    "fwd_date": hit.value_date})
    return pd.DataFrame(out)
