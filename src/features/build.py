"""Build the modelling panel: FBref season stats joined to Transfermarkt labels.

Alignment: stats from season t predict the Transfermarkt valuation published at
the start of season t+1. Nothing dated on or after the label enters X.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.entity.resolve import COMP_MAP
from src.features.definitions import COUNTS, POSITION_MAP, RENAME, VOLUME, WEIGHTED
from src.features.labels import attach_real_labels

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
KEY = ["Season_End_Year", "Squad", "Comp", "Url"]
SEASONS = range(2018, 2024)          # FBref Season_End_Year
MIN_MINUTES = 600


# --------------------------------------------------------------------------- #
# 1. wide FBref matrix
# --------------------------------------------------------------------------- #
def load_fbref_wide() -> pd.DataFrame:
    """Merge the nine FBref stat tables into one player-squad-season row."""
    base = pd.read_parquet(RAW / "fbref_standard.parquet")
    base = base[base.Season_End_Year.isin(SEASONS)]
    base = base[base.Url.str.contains(r"/players/[0-9a-f]+/", na=False)]

    keep = KEY + ["Player", "Nation", "Pos", "Born"]
    keep += COUNTS["standard"] + VOLUME["standard"]
    wide = base[keep].copy()

    for table, cols in COUNTS.items():
        if table == "standard":
            continue
        df = pd.read_parquet(RAW / f"fbref_{table}.parquet")
        df = df[df.Season_End_Year.isin(SEASONS)]
        take = [c for c in cols + WEIGHTED.get(table, []) if c in df.columns]
        missing = set(cols) - set(df.columns)
        if missing:
            raise KeyError(f"{table} missing {missing}")
        df = df[KEY + take].drop_duplicates(KEY)
        wide = wide.merge(df, on=KEY, how="left", suffixes=("", f"_{table}"))
    return wide


# --------------------------------------------------------------------------- #
# 2. collapse mid-season transfers
# --------------------------------------------------------------------------- #
def aggregate_player_season(wide: pd.DataFrame) -> pd.DataFrame:
    """One row per (player, season). Counting stats sum; rates weight by minutes."""
    count_cols = [c for cols in COUNTS.values() for c in cols]
    vol_cols = VOLUME["standard"]
    wt_cols = [c for cols in WEIGHTED.values() for c in cols if c in wide.columns]

    wide = wide.sort_values("Min_Playing", ascending=False)
    g = wide.groupby(["Url", "Season_End_Year"], sort=False)

    out = g[count_cols + vol_cols].sum(min_count=1)

    # Minutes-weighted means for the rate-like columns.
    for c in wt_cols:
        num = wide[c].mul(wide.Min_Playing)
        out[c] = (num.groupby([wide.Url, wide.Season_End_Year]).sum()
                  / g.Min_Playing.sum())

    # Context comes from the club the player spent most minutes at.
    first = g[["Player", "Nation", "Pos", "Born", "Squad", "Comp"]].first()
    out = out.join(first)
    out["n_clubs"] = g.size()
    return out.reset_index()


# --------------------------------------------------------------------------- #
# 3. per-90 rates and ratios
# --------------------------------------------------------------------------- #
def derive_rates(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns=RENAME)
    per90 = df.minutes / 90.0

    rate_src = [RENAME[c] for cols in COUNTS.values() for c in cols]
    for c in rate_src:
        df[f"{c}_p90"] = df[c] / per90

    with np.errstate(divide="ignore", invalid="ignore"):
        df["shot_accuracy"] = df.shots_on_target / df.shots.replace(0, np.nan)
        df["pass_cmp_pct"] = df.passes_cmp / df.passes_att.replace(0, np.nan)
        df["long_cmp_pct"] = df.long_cmp / df.long_att.replace(0, np.nan)
        df["dribble_succ_pct"] = df.dribbles_succ / df.dribbles_att.replace(0, np.nan)
        df["aerial_win_pct"] = df.aerials_won / (
            df.aerials_won + df.aerials_lost).replace(0, np.nan)
        df["npxg_per_shot"] = df.npxg / df.shots.replace(0, np.nan)
        df["tackle_win_pct"] = df.tackles_won / df.tackles.replace(0, np.nan)

    # Overperformance: the regression-to-mean signal.
    df["g_minus_xg"] = df.np_goals - df.npxg
    df["a_minus_xag"] = df.assists - df.xag
    df["g_minus_xg_p90"] = df.g_minus_xg / per90
    df["a_minus_xag_p90"] = df.a_minus_xag / per90
    df["npxg_xag_p90"] = (df.npxg + df.xag) / per90
    df["ga_p90"] = (df.goals + df.assists) / per90

    df["primary_pos"] = df.Pos.str.split(",").str[0]
    df["pos_count"] = df.Pos.str.count(",").add(1)
    return df


# --------------------------------------------------------------------------- #
# 4. Transfermarkt: labels, bio, contract, prior value
# --------------------------------------------------------------------------- #
LAG_FEATURES = [
    "minutes", "starts", "npxg_xag_p90", "ga_p90", "goals_p90", "assists_p90",
    "sca_p90", "prog_passes_p90", "tackles_p90", "touches_p90", "minutes_pct",
]


STALE_THRESHOLD = 0.90
MIN_MATCHES_FOR_COMPLETE_SEASON = 30      # a Big 5 league plays 34-38


def _tm_by_season() -> pd.DataFrame:
    """One Transfermarkt row per (player, season); mid-season movers appear twice."""
    tm = pd.read_parquet(RAW / "tm_vals.parquet")
    tm = tm.sort_values("player_market_value_euro", ascending=False)
    return tm.drop_duplicates(["player_url", "season_start_year"], keep="first")


def partial_seasons(df: pd.DataFrame) -> set[int]:
    """Seasons the source only partly covers.

    The mirror stopped updating in November 2022, so its 2022-23 rows hold about
    thirteen matches. Those stats cannot sit in the same panel as full seasons:
    per-90 rates are far noisier and every volume feature is on a different
    scale, while the label is still a full post-season valuation.
    """
    played = df.groupby("Season_End_Year").matches.max()
    return {int(y) for y, m in played.items()
            if pd.notna(m) and m < MIN_MATCHES_FOR_COMPLETE_SEASON}


def stale_label_seasons(threshold: float = STALE_THRESHOLD) -> set[int]:
    """Snapshots that merely copy the previous season are unusable as labels.

    Transfermarkt normally revalues 78-95% of players between seasons. A snapshot
    that leaves nearly everyone unchanged has not been refreshed, and using it as
    a label makes the carry-forward baseline score perfectly while teaching the
    model nothing.
    """
    tm = _tm_by_season()
    stale = set()
    for year in sorted(tm.season_start_year.unique()):
        prev = tm[tm.season_start_year == year - 1].set_index(
            "player_url").player_market_value_euro
        cur = tm[tm.season_start_year == year].set_index(
            "player_url").player_market_value_euro
        both = prev.index.intersection(cur.index)
        if len(both) < 200:
            continue
        if (prev.loc[both] == cur.loc[both]).mean() >= threshold:
            stale.add(int(year))
    return stale


def _fill_bio_from_player_history(df: pd.DataFrame, tm: pd.DataFrame) -> pd.DataFrame:
    """Backfill static bio fields from any season in which the mirror saw the player."""
    def _mode(s: pd.Series):
        m = s.dropna().mode()
        return m.iloc[0] if len(m) else None

    per_player = tm.groupby("player_url").agg(
        _pos=("player_position", _mode),
        _nat=("player_nationality", _mode),
        _foot=("player_foot", _mode),
        _height=("player_height_mtrs", _mode),
        _dob=("player_dob", _mode),
    )
    # Squad is season-specific, so take the club recorded closest to that season.
    squad = (tm.dropna(subset=["squad"])
               .sort_values("season_start_year")
               .groupby("player_url").squad.last().rename("_squad"))
    per_player = per_player.join(squad)

    j = df.tm_url.map(per_player.to_dict("index"))
    for col, key in [("player_position", "_pos"), ("player_nationality", "_nat"),
                     ("player_foot", "_foot"), ("player_height_mtrs", "_height"),
                     ("player_dob", "_dob"), ("tm_squad", "_squad")]:
        fill = j.map(lambda d, k=key: d.get(k) if isinstance(d, dict) else None)
        df[col] = df[col].where(df[col].notna(), fill)
    return df


def attach_transfermarkt(df: pd.DataFrame) -> pd.DataFrame:
    link = pd.read_parquet(ROOT / "data/interim/big5_resolution.parquet")
    link = link.dropna(subset=["tm_url"]).drop_duplicates(["fb_url", "season_end_year"])
    df = df.merge(link[["fb_url", "season_end_year", "tm_url", "tier"]],
                  left_on=["Url", "Season_End_Year"],
                  right_on=["fb_url", "season_end_year"], how="left")

    tm = _tm_by_season()

    # Label: value published at the start of season t+1.
    label = tm[["player_url", "season_start_year", "player_market_value_euro"]].rename(
        columns={"player_market_value_euro": "value_eur"})
    df = df.merge(label, left_on=["tm_url", "Season_End_Year"],
                  right_on=["player_url", "season_start_year"], how="left") \
           .drop(columns=["player_url", "season_start_year"])

    # Prior value: published before season t. Feature for the `update` variant only.
    prior = label.rename(columns={"value_eur": "prior_value_eur"})
    prior["_join_season"] = prior.season_start_year + 1
    df = df.merge(prior[["player_url", "_join_season", "prior_value_eur"]],
                  left_on=["tm_url", "Season_End_Year"],
                  right_on=["player_url", "_join_season"], how="left") \
           .drop(columns=["player_url", "_join_season"])

    # Bio and contract, taken from the row contemporaneous with the label.
    bio = tm[["player_url", "season_start_year", "player_dob", "player_position",
              "player_nationality", "player_height_mtrs", "player_foot",
              "contract_expiry", "date_joined", "squad"]].rename(
        columns={"squad": "tm_squad"})
    df = df.merge(bio, left_on=["tm_url", "Season_End_Year"],
                  right_on=["player_url", "season_start_year"], how="left") \
           .drop(columns=["player_url", "season_start_year"])

    # Scraped labels admit rows the mirror never covered in that season, leaving
    # bio fields null. A player's position, height and foot do not change, so
    # fall back to his most common values across every season the mirror has.
    df = _fill_bio_from_player_history(df, tm)

    # The label is dated at the start of season t+1; derive age/contract from it.
    label_date = pd.to_datetime(dict(year=df.Season_End_Year, month=7, day=1))
    dob = pd.to_datetime(df.player_dob, errors="coerce")
    df["age"] = (label_date - dob).dt.days / 365.25
    exp = pd.to_datetime(df.contract_expiry, errors="coerce")
    df["contract_months_left"] = (exp - label_date).dt.days / 30.44
    joined = pd.to_datetime(df.date_joined, errors="coerce")
    df["years_at_club"] = (label_date - joined).dt.days / 365.25
    df["height_m"] = pd.to_numeric(df.player_height_mtrs, errors="coerce")
    return df


# --------------------------------------------------------------------------- #
# 5. context and lags
# --------------------------------------------------------------------------- #
def add_context(df: pd.DataFrame) -> pd.DataFrame:
    """Squad wealth, league strength, and the inflation deflator.

    All three are taken from the snapshot *before* season t. The contemporaneous
    snapshot is the one the label comes from, so a squad total or league median
    built on it would contain the player's own label - aggregate leakage that a
    correlation check will not catch.
    """
    df = df.copy()
    tm = _tm_by_season()
    tm = tm[tm.player_market_value_euro.notna()]

    squad_val = (tm.groupby(["comp_name", "season_start_year", "squad"])
                   .player_market_value_euro.sum().rename("squad_value_prior")
                   .reset_index())
    squad_val["squad_value_rank_prior"] = (
        squad_val.groupby(["comp_name", "season_start_year"])
                 .squad_value_prior.rank(ascending=False))
    squad_val["_join_season"] = squad_val.season_start_year + 1

    df["_comp_tm"] = df.Comp.map(COMP_MAP)
    df = df.merge(
        squad_val[["comp_name", "_join_season", "squad",
                   "squad_value_prior", "squad_value_rank_prior"]],
        left_on=["_comp_tm", "Season_End_Year", "tm_squad"],
        right_on=["comp_name", "_join_season", "squad"], how="left") \
        .drop(columns=["comp_name", "_join_season", "squad"])

    league = (tm.groupby(["comp_name", "season_start_year"])
                .player_market_value_euro.median().rename("league_median_prior")
                .reset_index())
    league["_join_season"] = league.season_start_year + 1
    df = df.merge(league[["comp_name", "_join_season", "league_median_prior"]],
                  left_on=["_comp_tm", "Season_End_Year"],
                  right_on=["comp_name", "_join_season"], how="left") \
           .drop(columns=["comp_name", "_join_season", "_comp_tm"])

    # Deflate by the prior-season median so the model learns quality, not
    # inflation - and so the deflator is knowable at prediction time.
    df["value_deflated"] = df.value_eur / df.league_median_prior
    df["prior_value_deflated"] = df.prior_value_eur / df.league_median_prior
    df["log_value"] = np.log1p(df.value_eur)
    df["log_value_deflated"] = np.log1p(df.value_deflated)
    df["squad_value_share"] = df.prior_value_eur / df.squad_value_prior
    return df


def add_lags(df: pd.DataFrame) -> pd.DataFrame:
    """True season-1 and season-2 lags; null when the player was absent."""
    df = df.copy()
    for k in (1, 2):
        lag = df[["Url", "Season_End_Year"] + LAG_FEATURES].copy()
        lag["Season_End_Year"] += k
        lag = lag.rename(columns={c: f"{c}_lag{k}" for c in LAG_FEATURES})
        df = df.merge(lag, on=["Url", "Season_End_Year"], how="left")
    df["minutes_growth"] = df.minutes - df.minutes_lag1
    df["npxg_xag_growth"] = df.npxg_xag_p90 - df.npxg_xag_p90_lag1
    df["seasons_observed"] = df[["minutes_lag1", "minutes_lag2"]].notna().sum(axis=1)
    return df


def build_panel() -> pd.DataFrame:
    df = derive_rates(aggregate_player_season(load_fbref_wide()))
    df = attach_transfermarkt(df)
    # Scraped, dated valuations supersede the mirror's season snapshots where
    # available - they repair the stale 2022 snapshot and add 2022-23.
    df = attach_real_labels(df, list(SEASONS))
    df = add_context(df)
    df = add_lags(df)
    df = df.copy()
    df["pos_group"] = df.player_position.map(POSITION_MAP)
    # A stale mirror snapshot is only a problem for rows still sourced from it.
    stale = stale_label_seasons()
    df["label_is_stale"] = (df.Season_End_Year.isin(stale)
                            & (df.get("label_source", "mirror") == "mirror"))
    df["season_is_partial"] = df.Season_End_Year.isin(partial_seasons(df))
    df["eligible"] = (
        (df.minutes >= MIN_MINUTES)
        & (df.primary_pos != "GK")
        & (df.player_position != "Goalkeeper")
        & ~df.label_is_stale
        & ~df.season_is_partial
        # A zero valuation is an absence of data, not a price.
        & (df.value_eur > 0)
        & df.pos_group.notna()
    )
    return df.copy()          # de-fragment after the many assignments
