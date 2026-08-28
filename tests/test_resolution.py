"""The match-rate gate. P1 must not proceed on a worse join than P0 proved."""
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.entity.resolve import norm_club, norm_name  # noqa: E402

GATE = 95.0
MIN_MINUTES = 600


@pytest.fixture(scope="module")
def matched() -> pd.DataFrame:
    link = pd.read_parquet(ROOT / "data/interim/big5_resolution.parquet")
    fb = pd.read_parquet(ROOT / "data/raw/fbref_standard.parquet")
    fb = fb[fb.Season_End_Year.between(2018, 2023)]
    m = link.merge(
        fb[["Season_End_Year", "Player", "Squad", "Min_Playing", "Pos"]],
        left_on=["season_end_year", "fb_player", "fb_squad"],
        right_on=["Season_End_Year", "Player", "Squad"], how="left",
    )
    m["matched"] = m.tm_url.notna()
    return m[(m.Min_Playing >= MIN_MINUTES) & (m.Pos != "GK")]


def test_overall_match_rate(matched):
    rate = matched.matched.mean() * 100
    assert rate >= GATE, f"Big 5 match rate {rate:.1f}% below {GATE}% gate"


def test_per_league_match_rate(matched):
    by = matched.groupby("comp").matched.mean() * 100
    bad = by[by < GATE]
    assert bad.empty, f"leagues below {GATE}% gate:\n{bad.round(1).to_string()}"


def test_no_duplicate_links(matched):
    """One FBref player-season must not resolve to two Transfermarkt players."""
    key = ["season_end_year", "fb_player", "fb_squad"]
    assert not matched.duplicated(key).any()


@pytest.mark.parametrize("raw,want", [
    ("Rúben Dias", "ruben dias"),
    ("Pierre-Emerick Aubameyang", "pierre emerick aubameyang"),
    ("Vinícius Júnior", "vinicius"),
])
def test_norm_name(raw, want):
    assert norm_name(raw) == want


@pytest.mark.parametrize("raw,want", [
    ("AFC Bournemouth", "bournemouth"),
    ("Arsenal FC", "arsenal"),
    ("Bayer 04 Leverkusen", "bayer 04 leverkusen"),
])
def test_norm_club(raw, want):
    assert norm_club(raw) == want
