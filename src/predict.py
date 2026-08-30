"""Look up any player's valuation and see where the model disagrees with Transfermarkt.

    from src.predict import search, lookup
    search("haaland")
    v = lookup("Erling Haaland", 2022)
    print(v)          # headline comparison
    v.why            # SHAP attribution, in euros
    v.comparables    # five most similar players

Every prediction here is out-of-sample: the season was scored by a model trained
only on earlier seasons.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import joblib
import pandas as pd

from src.explain.shap_utils import Explainer
from src.explain.similarity import SimilarPlayers

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"


def _m(x: float) -> str:
    return f"EUR {x/1e6:,.1f}m"


@lru_cache(maxsize=1)
def _load():
    preds = pd.read_parquet(PROC / "predictions.parquet")
    sim = SimilarPlayers(preds)
    models, background = {}, None
    try:
        background = joblib.load(PROC / "models/background.joblib")
        for v in ("coldstart", "update"):
            models[v] = joblib.load(PROC / f"models/conformal_lgbm_{v}.joblib")
    except FileNotFoundError:
        pass
    return preds, sim, models, background


def search(name: str, limit: int = 12) -> pd.DataFrame:
    """Find players whose name contains `name`, with the seasons available."""
    preds, *_ = _load()
    hit = preds[preds.Player.str.contains(name, case=False, na=False)]
    return (hit[["Player", "Comp", "Season_End_Year", "pos_group", "value_eur"]]
            .sort_values(["Player", "Season_End_Year"]).head(limit)
            .reset_index(drop=True))


@dataclass
class Valuation:
    player: str
    season: int
    league: str
    position: str
    age: float
    minutes: int
    model_value: float
    low: float
    high: float
    tm_value: float
    variant: str
    why: pd.DataFrame
    comparables: pd.DataFrame

    @property
    def delta(self) -> float:
        return self.model_value - self.tm_value

    def __str__(self) -> str:
        verdict = "undervalued" if self.delta > 0 else "overvalued"
        pct = abs(self.delta) / self.tm_value * 100
        lines = [
            f"{self.player}  ·  {self.season-1}-{str(self.season)[2:]}  ·  "
            f"{self.league}  ·  {self.position}  ·  age {self.age:.1f}  ·  "
            f"{self.minutes:,.0f} min",
            "",
            f"  model ({self.variant}) : {_m(self.model_value)}   "
            f"[{_m(self.low)} – {_m(self.high)}]",
            f"  transfermarkt      : {_m(self.tm_value)}",
            f"  delta              : {_m(self.delta)}  "
            f"({pct:.0f}% {verdict} by the model)",
        ]
        if self.why is not None and len(self.why):
            lines += ["", "  why:"]
            for r in self.why.itertuples():
                sign = "+" if r.effect_pct >= 0 else ""
                if pd.isna(r.value):
                    shown = "n/a"
                elif abs(r.value) >= 1e6:
                    shown = f"{r.value/1e6:,.0f}m"
                else:
                    shown = f"{r.value:,.2f}"
                lines.append(f"    {r.feature:<26} {shown:>12}   "
                             f"{sign}{r.effect_pct:>6.1f}%")
        return "\n".join(lines)


def lookup(player: str, season: int | None = None,
           variant: str = "coldstart", top: int = 6) -> Valuation:
    """Valuation for one player-season. Defaults to the most recent season held."""
    preds, sim, models, background = _load()
    hit = preds[preds.Player.str.lower() == player.lower()]
    if hit.empty:
        hit = preds[preds.Player.str.contains(player, case=False, na=False)]
    if hit.empty:
        raise KeyError(f"no player matching {player!r} — try search({player!r})")
    if season is not None:
        hit = hit[hit.Season_End_Year == season]
        if hit.empty:
            raise KeyError(f"{player!r} has no row for season ending {season}")
    row = hit.sort_values("Season_End_Year").iloc[[-1]]
    idx = row.index[0]

    why = None
    if variant in models and background is not None:
        try:
            why = Explainer(models[variant].base, background).explain_row(row, top=top)
        except Exception:
            why = None

    return Valuation(
        player=row.Player.iloc[0], season=int(row.Season_End_Year.iloc[0]),
        league=row.Comp.iloc[0], position=row.pos_group.iloc[0],
        age=float(row.age.iloc[0]), minutes=int(row.minutes.iloc[0]),
        model_value=float(row[f"pred_{variant}"].iloc[0]),
        low=float(row[f"lo_{variant}"].iloc[0]),
        high=float(row[f"hi_{variant}"].iloc[0]),
        tm_value=float(row.value_eur.iloc[0]), variant=variant,
        why=why, comparables=sim.find(preds.index.get_loc(idx), k=5),
    )
