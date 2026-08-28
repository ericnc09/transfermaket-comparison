"""Temporal splits. Random splits would leak future valuations into the past."""
from __future__ import annotations

import pandas as pd
from sklearn.model_selection import GroupKFold

# Season_End_Year. 2022 is excluded: its Transfermarkt label snapshot is a
# 99.5% copy of 2021, so it cannot be used to score anything.
TRAIN = [2018, 2019]
VALID = [2020]
TEST = [2021]


def temporal_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    return (df[df.Season_End_Year.isin(TRAIN)].copy(),
            df[df.Season_End_Year.isin(VALID)].copy(),
            df[df.Season_End_Year.isin(TEST)].copy())


def grouped_folds(df: pd.DataFrame, n_splits: int = 5):
    """CV folds that never split one player across train and validation."""
    return GroupKFold(n_splits=n_splits).split(df, groups=df.tm_url)
