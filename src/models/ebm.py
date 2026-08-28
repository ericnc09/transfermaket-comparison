"""Explainable Boosting Machine.

Near-GBM accuracy with a shape function per feature, so the learned age curve
can be read straight off the model rather than inferred from SHAP. This is the
model that produces the headline figure.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.features.manifest import feature_columns
from src.models.gbm import TARGET, _reinflate


class EBM:
    def __init__(self, variant: str = "coldstart", **kw):
        self.variant = variant
        self.name = f"EBM [{variant}]"
        self.kw = dict(interactions=10, outer_bags=8, inner_bags=0,
                       learning_rate=0.02, max_bins=256,
                       random_state=0) | kw

    def fit(self, train: pd.DataFrame, valid: pd.DataFrame | None = None):
        from interpret.glassbox import ExplainableBoostingRegressor
        self.feats_ = feature_columns(train, self.variant, require_variance=True)
        self.m_ = ExplainableBoostingRegressor(feature_names=self.feats_, **self.kw)
        self.m_.fit(train[self.feats_], train[TARGET])
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return _reinflate(self.m_.predict(df[self.feats_]), df)

    def shape_function(self, feature: str) -> pd.DataFrame:
        """Bin edges and the model's additive contribution - e.g. the age curve."""
        idx = self.m_.term_names_.index(feature)
        return pd.DataFrame({
            "bin": self.m_.bins_[idx][0] if self.m_.bins_[idx] else [],
            "contribution": np.asarray(self.m_.term_scores_[idx][1:-1], float),
        })
