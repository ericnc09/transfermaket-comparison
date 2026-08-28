"""Leakage and integrity gates on the modelling panel.

Every test here answers one question: could the model see, at training time,
information that would not exist at prediction time?
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.features.build import MIN_MINUTES  # noqa: E402
from src.features.manifest import PRIOR_VALUE, TARGETS, feature_columns  # noqa: E402

PROC = ROOT / "data" / "processed"


@pytest.fixture(scope="module")
def panel() -> pd.DataFrame:
    return pd.read_parquet(PROC / "panel_model.parquet")


@pytest.fixture(scope="module")
def manifest() -> dict:
    return json.loads((PROC / "panel_manifest.json").read_text())


# --------------------------------------------------------------------------- #
# integrity
# --------------------------------------------------------------------------- #
def test_one_row_per_player_season(panel):
    assert not panel.duplicated(["Url", "Season_End_Year"]).any()


def test_eligibility_filter_applied(panel):
    assert (panel.minutes >= MIN_MINUTES).all()
    assert (panel.primary_pos != "GK").all()
    assert (panel.player_position != "Goalkeeper").all()


def test_every_row_has_a_label(panel):
    assert panel.value_eur.notna().all()
    assert (panel.value_eur > 0).all()


def test_position_groups_complete(panel):
    assert panel.pos_group.notna().all()
    assert set(panel.pos_group) == {"CB", "FB", "DM", "CM", "AM", "W", "ST"}


# --------------------------------------------------------------------------- #
# leakage
# --------------------------------------------------------------------------- #
def test_no_feature_is_a_transform_of_the_target(panel, manifest):
    """A >0.98 correlation with the target means the label leaked in."""
    y = panel.log_value
    offenders = {}
    for c in manifest["features_update"]:
        v = panel[c]
        if v.notna().sum() < 100 or v.nunique() < 3:
            continue
        r = np.corrcoef(v.fillna(v.median()), y)[0, 1]
        if abs(r) > 0.98:
            offenders[c] = round(float(r), 4)
    assert not offenders, f"features leaking the target: {offenders}"


def test_coldstart_excludes_prior_valuation(manifest):
    assert not PRIOR_VALUE & set(manifest["features_coldstart"])
    assert PRIOR_VALUE <= set(manifest["features_update"])


def test_no_target_column_is_a_feature(manifest):
    for variant in ("features_coldstart", "features_update"):
        assert not TARGETS & set(manifest[variant])


def test_prior_value_precedes_the_label(panel):
    """prior_value must come from season t-1, the label from season t+1."""
    tm = pd.read_parquet(ROOT / "data/raw/tm_vals.parquet")
    tm = tm.sort_values("player_market_value_euro", ascending=False) \
           .drop_duplicates(["player_url", "season_start_year"])
    lookup = tm.set_index(["player_url", "season_start_year"]).player_market_value_euro

    sample = panel.dropna(subset=["prior_value_eur"]).sample(300, random_state=0)
    for r in sample.itertuples():
        assert lookup.get((r.tm_url, r.Season_End_Year)) == r.value_eur
        assert lookup.get((r.tm_url, r.Season_End_Year - 1)) == r.prior_value_eur


def test_lags_are_strictly_backward(panel):
    """A lag must equal the same player's value one season earlier.

    Lags are drawn from the full panel, not the filtered one: a player whose
    previous season fell below the minutes bar still has a real lag, and that
    breakout signal is exactly what the feature is for.
    """
    full = pd.read_parquet(PROC / "panel_full.parquet")
    cur = full.set_index(["Url", "Season_End_Year"]).minutes
    sample = panel.dropna(subset=["minutes_lag1"]).sample(300, random_state=0)
    for r in sample.itertuples():
        assert cur.get((r.Url, r.Season_End_Year - 1)) == r.minutes_lag1


def test_lag_coverage_is_plausible(panel):
    """Roughly a third of rows are a player's first observed season."""
    have = panel.minutes_lag1.notna().mean()
    assert 0.55 < have < 0.80, f"lag1 coverage {have:.1%} outside expected band"


def test_discontinued_stats_absent_only_outside_training(manifest):
    """Pressures and progressive carries vanish in 2022-23, which carries no label."""
    full = pd.read_parquet(PROC / "panel_full.parquet")
    labelled_seasons = set(manifest["seasons"])
    assert 2023 not in labelled_seasons
    for col in ("pressures_p90", "prog_carries_p90"):
        in_train = full[full.Season_End_Year.isin(labelled_seasons) & full.eligible]
        assert in_train[col].notna().mean() > 0.95


def test_context_aggregates_predate_the_label(panel):
    """Squad and league aggregates must come from the snapshot BEFORE season t.

    Built on the contemporaneous snapshot they would contain the player's own
    label. Correlation checks miss this: the leak is diluted across a whole
    squad, so it shows up as r≈0.03 rather than r≈1.
    """
    tm = pd.read_parquet(ROOT / "data/raw/tm_vals.parquet")
    tm = tm.sort_values("player_market_value_euro", ascending=False) \
           .drop_duplicates(["player_url", "season_start_year"])
    tm = tm[tm.player_market_value_euro.notna()]

    squad_tot = tm.groupby(["squad", "season_start_year"]).player_market_value_euro.sum()
    league_med = tm.groupby(["comp_name", "season_start_year"]).player_market_value_euro.median()

    sample = panel.dropna(subset=["squad_value_prior"]).sample(200, random_state=0)
    for r in sample.itertuples():
        prior = squad_tot.get((r.tm_squad, r.Season_End_Year - 1))
        assert prior == pytest.approx(r.squad_value_prior), (
            f"{r.Player}: squad_value_prior is not the season-{r.Season_End_Year - 1} total"
        )
        # And it must NOT equal the contemporaneous total.
        same = squad_tot.get((r.tm_squad, r.Season_End_Year))
        if same is not None and same != prior:
            assert r.squad_value_prior != same

    comp_map = {"Premier League": "Premier League", "La Liga": "LaLiga",
                "Serie A": "Serie A", "Bundesliga": "Bundesliga", "Ligue 1": "Ligue 1"}
    for r in panel.sample(100, random_state=1).itertuples():
        want = league_med.get((comp_map[r.Comp], r.Season_End_Year - 1))
        assert want == pytest.approx(r.league_median_prior)


def test_stale_label_seasons_are_excluded(panel):
    """A snapshot that merely copies the previous season cannot serve as a label.

    Transfermarkt's 2022 snapshot is a 99.5% copy of 2021. Left in, it lets the
    carry-forward baseline score a perfect R2 and makes every model look good
    for the wrong reason.
    """
    from src.features.build import stale_label_seasons

    stale = stale_label_seasons()
    assert 2022 in stale, "the known-stale 2022 snapshot is no longer detected"
    assert not set(panel.Season_End_Year) & stale


def test_label_moves_between_seasons(panel):
    """Sanity floor: labels must actually differ from the prior valuation."""
    moved = (panel.prior_value_eur != panel.value_eur).mean()
    assert moved > 0.5, (
        f"only {moved:.1%} of labels differ from their prior value - "
        "the label snapshot is probably stale"
    )
