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


def rolling_origin(df: pd.DataFrame, min_train_seasons: int = 1):
    """Expanding-window evaluation: train on all seasons before t, test on t.

    With only four label-able seasons a single train/valid/test split spends most
    of the panel on a two-season training window and reports one number from one
    season. Rolling origin uses every season as a test fold in turn, which is the
    standard protocol for a short panel and gives a variance estimate as well as
    a point estimate.
    """
    seasons = sorted(df.Season_End_Year.unique())
    for i in range(min_train_seasons, len(seasons)):
        test_season = seasons[i]
        train = df[df.Season_End_Year < test_season]
        test = df[df.Season_End_Year == test_season]
        if len(train) and len(test):
            yield int(test_season), train.copy(), test.copy()
