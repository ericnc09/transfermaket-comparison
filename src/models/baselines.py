"""Tier 0 baselines. Every leaderboard reports these alongside any real model.

The carry-forward baseline in particular is brutally strong: a headline R2 that
is not shown next to it is not interpretable.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


class Baseline:
    name = "baseline"

    def fit(self, train: pd.DataFrame, valid: pd.DataFrame | None = None):
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:  # euros
        raise NotImplementedError


class GlobalMedian(Baseline):
    name = "global median"

    def fit(self, train, valid=None):
        self.v_ = float(train.value_eur.median())
        return self

    def predict(self, df):
        return np.full(len(df), self.v_)


class PositionAgeMedian(Baseline):
    """Median value within (position group x 2-year age bucket)."""
    name = "median by position x age"

    @staticmethod
    def _bucket(df):
        return pd.cut(df.age, bins=range(16, 42, 2))

    def fit(self, train, valid=None):
        t = train.assign(_b=self._bucket(train))
        self.tbl_ = t.groupby(["pos_group", "_b"], observed=True).value_eur.median()
        self.fallback_ = float(train.value_eur.median())
        return self

    def predict(self, df):
        idx = pd.MultiIndex.from_arrays([df.pos_group, self._bucket(df)])
        return self.tbl_.reindex(idx).fillna(self.fallback_).to_numpy()


class LogLinear(Baseline):
    """OLS on log value: age, age^2, minutes, goal involvement."""
    name = "log-linear (age, minutes, G+A)"
    COLS = ["age", "minutes", "ga_p90"]

    def _design(self, df):
        X = df[self.COLS].to_numpy(float)
        X = np.column_stack([X, df.age.to_numpy(float) ** 2])
        return np.nan_to_num(X, nan=0.0)

    def fit(self, train, valid=None):
        from sklearn.linear_model import LinearRegression
        self.m_ = LinearRegression().fit(self._design(train), np.log1p(train.value_eur))
        return self

    def predict(self, df):
        return np.expm1(self.m_.predict(self._design(df)))


class CarryForward(Baseline):
    """Last season's Transfermarkt value, unchanged. The baseline that matters."""
    name = "carry forward prior TM value"

    def fit(self, train, valid=None):
        self.fallback_ = float(train.value_eur.median())
        return self

    def predict(self, df):
        return df.prior_value_eur.fillna(self.fallback_).to_numpy()


ALL = [GlobalMedian, PositionAgeMedian, LogLinear, CarryForward]
