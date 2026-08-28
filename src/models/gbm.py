"""Gradient boosting on the deflated log target.

Target is log1p(value / league_median_prior): deflating by the *prior* season's
median removes market inflation using a quantity knowable at prediction time.
Predictions are re-inflated before scoring, so every model is compared in euros.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.features.manifest import CATEGORICAL, feature_columns

TARGET = "log_value_deflated"


def _reinflate(pred_log: np.ndarray, df: pd.DataFrame) -> np.ndarray:
    return np.expm1(pred_log) * df.league_median_prior.to_numpy()


class HistGBM:
    """sklearn's histogram gradient booster - the LightGBM algorithm, no OpenMP."""

    def __init__(self, variant: str = "coldstart", **kw):
        self.variant = variant
        self.name = f"HistGBM [{variant}]"
        self.kw = dict(max_iter=600, learning_rate=0.05, max_leaf_nodes=31,
                       min_samples_leaf=20, l2_regularization=1.0,
                       early_stopping=True, n_iter_no_change=50,
                       validation_fraction=0.15, random_state=0) | kw

    def fit(self, train: pd.DataFrame, valid: pd.DataFrame | None = None):
        from sklearn.ensemble import HistGradientBoostingRegressor
        self.feats_ = feature_columns(train, self.variant, require_variance=True)
        self.m_ = HistGradientBoostingRegressor(**self.kw)
        self.m_.fit(train[self.feats_], train[TARGET])
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return _reinflate(self.m_.predict(df[self.feats_]), df)


class CatBoost:
    """CatBoost - handles club/league/foot categoricals natively."""

    def __init__(self, variant: str = "coldstart", **kw):
        self.variant = variant
        self.name = f"CatBoost [{variant}]"
        self.kw = dict(iterations=2000, learning_rate=0.04, depth=6,
                       l2_leaf_reg=3.0, loss_function="RMSE",
                       random_seed=0, verbose=False) | kw

    def _frame(self, df: pd.DataFrame) -> pd.DataFrame:
        X = df[self.feats_ + CATEGORICAL].copy()
        for c in CATEGORICAL:
            X[c] = X[c].astype("object").fillna("unknown").astype(str)
        return X

    def fit(self, train: pd.DataFrame, valid: pd.DataFrame | None = None):
        from catboost import CatBoostRegressor, Pool
        self.feats_ = feature_columns(train, self.variant, require_variance=True)
        tr = Pool(self._frame(train), train[TARGET], cat_features=CATEGORICAL)
        ev = (Pool(self._frame(valid), valid[TARGET], cat_features=CATEGORICAL)
              if valid is not None else None)
        self.m_ = CatBoostRegressor(**self.kw)
        self.m_.fit(tr, eval_set=ev, early_stopping_rounds=100, verbose=False)
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return _reinflate(self.m_.predict(self._frame(df)), df)
