"""SHAP attributions for a single valuation.

Answers the question the residual leaderboard raises: the model says this player
is worth EUR 12m more than Transfermarkt does - on what grounds?
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# Feature names as they should read to a human.
PRETTY = {
    "age": "age", "minutes": "minutes played", "minutes_pct": "share of team minutes",
    "npxg_xag_p90": "npxG + xAG /90", "ga_p90": "goals + assists /90",
    "goals_p90": "goals /90", "assists_p90": "assists /90",
    "contract_months_left": "months left on contract",
    "squad_value_prior": "squad wealth", "squad_value_rank_prior": "squad rank in league",
    "league_median_prior": "league strength", "prior_value_eur": "previous TM value",
    "sca_p90": "shot-creating actions /90", "prog_carries_p90": "progressive carries /90",
    "prog_passes_p90": "progressive passes /90", "tackles_p90": "tackles /90",
    "aerial_win_pct": "aerial win %", "years_at_club": "years at club",
    "g_minus_xg": "goals above xG", "starts": "starts",
}


def pretty(name: str) -> str:
    return PRETTY.get(name, name.replace("_", " "))


class Explainer:
    """Wraps a fitted tree model and returns per-row euro attributions."""

    def __init__(self, model, background: pd.DataFrame):
        import shap

        self.model = model
        self.feats = model.feats_
        inner = getattr(model, "m_", model)
        self.explainer = shap.TreeExplainer(inner)
        self.background = background

    def explain_row(self, row: pd.DataFrame, top: int = 8) -> pd.DataFrame:
        """Top contributions for one row, as multiplicative effects on value.

        SHAP values are additive on the model's log target, so each one is a
        *multiplier* on euros, not a fixed euro amount. Reporting them as euros
        would require walking the prediction feature by feature, which makes the
        answer depend on the order chosen - two equally valid orderings give two
        different euro splits. The multiplier is order-independent and therefore
        the honest way to state it.
        """
        sv = np.asarray(self.explainer.shap_values(row[self.feats])).reshape(-1)
        order = np.argsort(-np.abs(sv))[:top]
        return pd.DataFrame([{
            "feature": pretty(self.feats[i]),
            "value": row[self.feats].iloc[0, i],
            "shap_log": float(sv[i]),
            "effect_x": float(np.exp(sv[i])),
            "effect_pct": float((np.exp(sv[i]) - 1) * 100),
        } for i in order])

    def global_importance(self, sample: pd.DataFrame, n: int = 400) -> pd.DataFrame:
        s = sample.sample(min(n, len(sample)), random_state=0)
        sv = np.asarray(self.explainer.shap_values(s[self.feats]))
        imp = np.abs(sv).mean(axis=0)
        return (pd.DataFrame({"feature": [pretty(f) for f in self.feats],
                              "mean_abs_shap": imp})
                .sort_values("mean_abs_shap", ascending=False)
                .reset_index(drop=True))
