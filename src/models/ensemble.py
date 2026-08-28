"""Stacked ensemble: a ridge meta-learner over out-of-fold base predictions."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.eval.splits import grouped_folds
from src.models.gbm import TARGET, _reinflate


def _to_target_space(pred_eur: np.ndarray, df: pd.DataFrame) -> np.ndarray:
    """Euros -> the deflated log space the meta-learner is trained in.

    Without this the stacker mixes units: base predictions arrive as log euros
    while the meta target is log(value / league_median_prior), so the ridge has
    to absorb a per-row deflator it cannot see.
    """
    return np.log1p(np.clip(pred_eur, 1e4, None)
                    / df.league_median_prior.to_numpy())


class Stacked:
    def __init__(self, bases: list, name: str = "Stacked", n_splits: int = 4):
        self.bases = bases
        self.name = name
        self.n_splits = n_splits

    def fit(self, train: pd.DataFrame, valid: pd.DataFrame | None = None):
        from sklearn.linear_model import RidgeCV

        oof = np.full((len(train), len(self.bases)), np.nan)
        pos = {ix: i for i, ix in enumerate(train.index)}
        for tr_i, va_i in grouped_folds(train, self.n_splits):
            tr, va = train.iloc[tr_i], train.iloc[va_i]
            rows = [pos[ix] for ix in va.index]
            for j, proto in enumerate(self.bases):
                m = proto.__class__(proto.variant)
                m.fit(tr, None)
                oof[rows, j] = _to_target_space(m.predict(va), va)

        ok = ~np.isnan(oof).any(axis=1)
        self.meta_ = RidgeCV(alphas=np.logspace(-3, 3, 20)).fit(
            oof[ok], train[TARGET].to_numpy()[ok])

        # Refit each base on the full training window for inference.
        self.fitted_ = []
        for proto in self.bases:
            m = proto.__class__(proto.variant)
            m.fit(train, valid)
            self.fitted_.append(m)
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        X = np.column_stack([_to_target_space(m.predict(df), df)
                             for m in self.fitted_])
        return _reinflate(self.meta_.predict(X), df)
