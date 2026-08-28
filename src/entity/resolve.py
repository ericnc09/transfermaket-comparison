"""Resolve FBref player-seasons against Transfermarkt squad rows.

The two sources share no key. We block on (season, competition), then run a
confidence cascade from exact name+birth-year agreement down to fuzzy name
matching, requiring an unambiguous winner at every tier.
"""
from __future__ import annotations

import re
import unicodedata

import pandas as pd
from rapidfuzz import fuzz, process
from unidecode import unidecode

# Club-name noise: legal suffixes and prefixes that differ between sources.
_CLUB_NOISE = re.compile(
    r"\b(fc|afc|cf|sc|ac|as|ss|ssc|us|ud|cd|rc|rcd|sd|ca|club|calcio|"
    r"association|athletic|atletico|deportivo|futbol|football|"
    r"borussia|fussball|sv|tsg|vfb|vfl|bsc|1899|1846|1900|1904|1907|1909|1919)\b",
    re.I,
)
_NAME_SUFFIX = re.compile(r"\b(jr|junior|sr|senior|ii|iii)\b", re.I)
_PUNCT = re.compile(r"[^a-z0-9 ]")
_WS = re.compile(r"\s+")


def norm_name(s: str | float) -> str:
    """Accent-fold, strip punctuation and generational suffixes, collapse space."""
    if not isinstance(s, str):
        return ""
    s = unidecode(unicodedata.normalize("NFKD", s)).lower()
    s = _PUNCT.sub(" ", s)
    s = _NAME_SUFFIX.sub(" ", s)
    return _WS.sub(" ", s).strip()


def norm_club(s: str | float) -> str:
    if not isinstance(s, str):
        return ""
    s = unidecode(unicodedata.normalize("NFKD", s)).lower()
    s = _PUNCT.sub(" ", s)
    s = _CLUB_NOISE.sub(" ", s)
    return _WS.sub(" ", s).strip()


def build_club_map(fb_clubs: list[str], tm_clubs: list[str], cutoff: int = 60) -> dict[str, str]:
    """Fuzzy crosswalk from FBref squad names to Transfermarkt squad names."""
    tm_norm = {norm_club(c): c for c in tm_clubs}
    keys = list(tm_norm)
    out: dict[str, str] = {}
    for c in fb_clubs:
        n = norm_club(c)
        if n in tm_norm:
            out[c] = tm_norm[n]
            continue
        hit = process.extractOne(n, keys, scorer=fuzz.token_set_ratio, score_cutoff=cutoff)
        if hit:
            out[c] = tm_norm[hit[0]]
    return out


def _unique_best(cands: pd.DataFrame, col: str = "score", margin: int = 5):
    """Return the single best candidate if it clears the runner-up by `margin`."""
    if cands.empty:
        return None
    s = cands.sort_values(col, ascending=False)
    if len(s) == 1 or s.iloc[0][col] - s.iloc[1][col] >= margin:
        return s.iloc[0]
    return None


def resolve_block(fb: pd.DataFrame, tm: pd.DataFrame, club_map: dict[str, str]) -> list[dict]:
    """Match one (season, competition) block. Returns one record per FBref row."""
    tm = tm.copy()
    tm["_name"] = tm.player_name.map(norm_name)
    tm["_by"] = tm.player_dob.map(lambda d: d.year if hasattr(d, "year") else None)
    tm_names = tm["_name"].tolist()

    out = []
    for row in fb.itertuples(index=False):
        name = norm_name(row.Player)
        by = int(row.Born) if pd.notna(row.Born) else None
        want_club = club_map.get(row.Squad)

        rec = {"fb_player": row.Player, "fb_squad": row.Squad, "fb_url": row.Url,
               "tm_url": None, "tier": None, "score": None}

        exact = tm[tm["_name"] == name]

        # Tier A - exact name and birth year agree.
        if by is not None and not exact.empty:
            hit = exact[exact["_by"] == by]
            if len(hit) == 1:
                out.append({**rec, "tm_url": hit.iloc[0].player_url, "tier": "A", "score": 100})
                continue
            if len(hit) > 1 and want_club is not None:
                hit2 = hit[hit.squad == want_club]
                if len(hit2) == 1:
                    out.append({**rec, "tm_url": hit2.iloc[0].player_url, "tier": "A", "score": 100})
                    continue

        # Tier B - exact name, unique within the block.
        if len(exact) == 1:
            out.append({**rec, "tm_url": exact.iloc[0].player_url, "tier": "B", "score": 100})
            continue
        if len(exact) > 1 and want_club is not None:
            hit = exact[exact.squad == want_club]
            if len(hit) == 1:
                out.append({**rec, "tm_url": hit.iloc[0].player_url, "tier": "B", "score": 100})
                continue

        # Tier C - birth year agrees, name is close.
        if by is not None:
            pool = tm[tm["_by"] == by]
            if not pool.empty:
                scored = pool.assign(
                    score=[fuzz.token_set_ratio(name, n) for n in pool["_name"]]
                )
                scored = scored[scored.score >= 80]
                if want_club is not None and (scored.squad == want_club).any():
                    scored = scored[scored.squad == want_club]
                best = _unique_best(scored)
                if best is not None:
                    out.append({**rec, "tm_url": best.player_url, "tier": "C",
                                "score": int(best.score)})
                    continue

        # Tier D - same club, name is close (catches missing/wrong birth years).
        if want_club is not None:
            pool = tm[tm.squad == want_club]
            if not pool.empty:
                scored = pool.assign(
                    score=[fuzz.token_set_ratio(name, n) for n in pool["_name"]]
                )
                scored = scored[scored.score >= 85]
                best = _unique_best(scored)
                if best is not None:
                    out.append({**rec, "tm_url": best.player_url, "tier": "D",
                                "score": int(best.score)})
                    continue

        # Tier E - high-confidence fuzzy anywhere in the block.
        scored = pd.DataFrame({
            "player_url": tm.player_url,
            "score": [fuzz.token_set_ratio(name, n) for n in tm_names],
        })
        scored = scored[scored.score >= 92]
        best = _unique_best(scored, margin=8)
        if best is not None:
            out.append({**rec, "tm_url": best.player_url, "tier": "E", "score": int(best.score)})
            continue

        out.append(rec)
    return out


def resolve(fb: pd.DataFrame, tm: pd.DataFrame) -> pd.DataFrame:
    """Resolve all (season, competition) blocks. FBref rows are the left side."""
    frames = []
    for (ey, comp), fb_blk in fb.groupby(["Season_End_Year", "Comp"]):
        tm_blk = tm[(tm.season_start_year == ey - 1) & (tm.comp_name == COMP_MAP[comp])]
        if tm_blk.empty:
            continue
        club_map = build_club_map(fb_blk.Squad.unique().tolist(),
                                  tm_blk.squad.unique().tolist())
        recs = resolve_block(fb_blk, tm_blk, club_map)
        frame = pd.DataFrame(recs)
        frame["season_end_year"] = ey
        frame["comp"] = comp
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


COMP_MAP = {
    "Premier League": "Premier League",
    "La Liga": "LaLiga",
    "Serie A": "Serie A",
    "Bundesliga": "Bundesliga",
    "Ligue 1": "Ligue 1",
}
