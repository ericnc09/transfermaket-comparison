"""Comparable players.

A nearest-neighbour lookup in standardised feature space. It is both a product
feature - "who does this player most resemble?" - and a sanity check: if the
neighbours of a EUR 80m winger are journeyman full-backs, the feature space is
not capturing what it should.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# Kept deliberately small and interpretable: style and role, not raw output.
SIM_FEATURES = [
    "age", "minutes_pct", "npxg_xag_p90", "ga_p90", "sca_p90",
    "prog_passes_p90", "prog_carries_p90", "tackles_p90", "interceptions_p90",
    "touches_att_pen_p90", "aerial_win_pct", "pass_cmp_pct", "dribbles_succ_p90",
]


class SimilarPlayers:
    def __init__(self, panel: pd.DataFrame, features: list[str] | None = None):
        self.features = [f for f in (features or SIM_FEATURES) if f in panel.columns]
        self.panel = panel.reset_index(drop=True)
        X = self.panel[self.features].to_numpy(float)
        med = np.nanmedian(X, axis=0)
        X = np.where(np.isnan(X), med, X)
        self.mu_, self.sd_ = X.mean(axis=0), X.std(axis=0) + 1e-9
        self.Z_ = (X - self.mu_) / self.sd_

    def find(self, idx: int, k: int = 5, same_position: bool = True) -> pd.DataFrame:
        d = np.linalg.norm(self.Z_ - self.Z_[idx], axis=1)
        mask = np.ones(len(self.panel), bool)
        mask[idx] = False
        # A player is not his own comparable in another season either.
        mask &= (self.panel.tm_url != self.panel.tm_url.iloc[idx]).to_numpy()
        if same_position:
            mask &= (self.panel.pos_group == self.panel.pos_group.iloc[idx]).to_numpy()
        cand = np.where(mask)[0]
        near = cand[np.argsort(d[cand])[:k]]
        out = self.panel.loc[near, ["Player", "Comp", "Season_End_Year",
                                    "pos_group", "age", "value_eur"]].copy()
        out["distance"] = d[near].round(3)
        return out.reset_index(drop=True)
