"""Regularised linear models with spline-transformed age.

Value falls off a cliff after about 27, so age enters through a natural spline
basis rather than as a raw term. These are the interpretable floor: if a
gradient booster cannot beat a splined ridge by a clear margin, the extra
complexity is not earning anything.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.features.manifest import feature_columns
from src.models.gbm import TARGET, _reinflate

AGE_COL = "age"


class _SplineLinear:
    _estimator = None

    def __init__(self, variant: str = "coldstart", **kw):
        self.variant = variant
        self.kw = kw

    def _pipeline(self, feats: list[str]):
        from sklearn.compose import ColumnTransformer
        from sklearn.impute import SimpleImputer
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import SplineTransformer, StandardScaler

        others = [c for c in feats if c != AGE_COL]
        age_pipe = Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("spline", SplineTransformer(n_knots=6, degree=3, include_bias=False)),
        ])
        rest_pipe = Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ])
        pre = ColumnTransformer([
            ("age", age_pipe, [AGE_COL] if AGE_COL in feats else []),
            ("rest", rest_pipe, others),
        ])
        return Pipeline([("pre", pre), ("model", self._estimator(**self.kw))])

    def fit(self, train: pd.DataFrame, valid: pd.DataFrame | None = None):
        self.feats_ = feature_columns(train, self.variant, require_variance=True)
        self.m_ = self._pipeline(self.feats_).fit(train[self.feats_], train[TARGET])
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return _reinflate(self.m_.predict(df[self.feats_]), df)


class RidgeSpline(_SplineLinear):
    def __init__(self, variant="coldstart", **kw):
        from sklearn.linear_model import RidgeCV
        self._estimator = RidgeCV
        super().__init__(variant, alphas=np.logspace(-2, 4, 25), **kw)
        self.name = f"Ridge+splines [{variant}]"


class ElasticNetSpline(_SplineLinear):
    def __init__(self, variant="coldstart", **kw):
        from sklearn.linear_model import ElasticNetCV
        self._estimator = ElasticNetCV
        super().__init__(variant, l1_ratio=[.1, .5, .9],
                         max_iter=5000, random_state=0, **kw)
        self.name = f"ElasticNet+splines [{variant}]"
