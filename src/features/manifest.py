"""Which columns are features, and which belong to which variant."""
from __future__ import annotations

import pandas as pd

# Never features: identifiers, raw source fields, and anything derived from the label.
IDENTIFIERS = {
    "Url", "fb_url", "tm_url", "Player", "Squad", "tm_squad", "Comp", "Nation",
    "Pos", "primary_pos", "player_dob", "player_position", "player_nationality",
    "player_height_mtrs", "contract_expiry", "date_joined", "tier", "eligible",
    "season_end_year", "fb_player", "fb_squad", "pos_group", "player_foot",
}
TARGETS = {"value_eur", "log_value", "value_deflated", "log_value_deflated"}

# Features that encode the previous Transfermarkt valuation.
PRIOR_VALUE = {"prior_value_eur", "prior_value_deflated", "squad_value_share"}

# Categoricals passed through to the model as categories, not dropped.
CATEGORICAL = ["pos_group", "player_foot", "Comp"]


def feature_columns(df: pd.DataFrame, variant: str = "coldstart") -> list[str]:
    """Numeric feature columns for a variant. `update` keeps the prior valuation."""
    if variant not in {"coldstart", "update"}:
        raise ValueError(variant)
    drop = IDENTIFIERS | TARGETS
    if variant == "coldstart":
        drop = drop | PRIOR_VALUE
    cols = [
        c for c in df.columns
        if c not in drop and pd.api.types.is_numeric_dtype(df[c])
    ]
    return sorted(cols)
