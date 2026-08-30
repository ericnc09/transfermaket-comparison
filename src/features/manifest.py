"""Which columns are features, and which belong to which variant."""
from __future__ import annotations

import pandas as pd

# Never features: identifiers, raw source fields, and anything derived from the label.
IDENTIFIERS = {
    "Url", "fb_url", "tm_url", "Player", "Squad", "tm_squad", "Comp", "Nation",
    "Pos", "primary_pos", "player_dob", "player_position", "player_nationality",
    "player_height_mtrs", "contract_expiry", "date_joined", "tier", "eligible",
    "season_end_year", "fb_player", "fb_squad", "pos_group", "player_foot",
    "label_is_stale", "season_is_partial", "label_source", "tm_player_id",
    # Season index must not be a feature: under a temporal split every test
    # season is later than anything in training, so a tree can only extrapolate
    # whatever the last training season taught it. Inflation is already handled
    # by deflating against league_median_prior.
    "Season_End_Year",
    # Birth year is age restated, and lets the model recover the season index.
    "Born",
    "label_date", "prior_date",
}
TARGETS = {"value_eur", "log_value", "value_deflated", "log_value_deflated"}

# Features that encode the previous Transfermarkt valuation.
PRIOR_VALUE = {"prior_value_eur", "prior_value_deflated", "squad_value_share"}

# Categoricals passed through to the model as categories, not dropped.
CATEGORICAL = ["pos_group", "player_foot", "Comp"]


def feature_columns(df: pd.DataFrame, variant: str = "coldstart",
                    require_variance: bool = False) -> list[str]:
    """Numeric feature columns for a variant. `update` keeps the prior valuation.

    With `require_variance`, columns that are constant or entirely null in `df`
    are dropped. Pass the training frame: a short training window can leave a
    lag column wholly absent, which some learners cannot bin.
    """
    if variant not in {"coldstart", "update"}:
        raise ValueError(variant)
    drop = IDENTIFIERS | TARGETS
    if variant == "coldstart":
        drop = drop | PRIOR_VALUE
    cols = [
        c for c in df.columns
        if c not in drop and pd.api.types.is_numeric_dtype(df[c])
    ]
    if require_variance:
        cols = [c for c in cols if df[c].nunique(dropna=True) >= 2]
    return sorted(cols)
