"""Prediction intervals.

A point estimate against a €7.7m median error in the top value quartile is false
precision. Two routes to a range:

* `QuantileTrio` fits three LightGBM models at p10/p50/p90 - cheap, but the
  coverage it delivers is whatever the fit happens to give.
* `Conformal` wraps any point model and calibrates on held-out residuals, giving
  a distribution-free coverage guarantee. Because the model works on a log
  target, the interval is multiplicative in euros, which is the right shape for
  value data - a €5m player and a €100m player do not carry the same absolute
  uncertainty.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.features.manifest import feature_columns
from src.models.gbm import TARGET, _reinflate


class QuantileTrio:
    """Three LightGBM quantile regressors: p10, p50, p90."""

    QUANTILES = (0.1, 0.5, 0.9)

    def __init__(self, variant: str = "coldstart", **kw):
        self.variant = variant
        self.name = f"LGBM-quantile [{variant}]"
        self.kw = dict(n_estimators=800, learning_rate=0.05, num_leaves=31,
                       min_child_samples=20, subsample=0.8, subsample_freq=1,
                       colsample_bytree=0.8, random_state=0, verbose=-1) | kw

    def fit(self, train: pd.DataFrame, valid: pd.DataFrame | None = None):
        import lightgbm as lgb
        self.feats_ = feature_columns(train, self.variant, require_variance=True)
        self.models_ = {}
        for q in self.QUANTILES:
            m = lgb.LGBMRegressor(objective="quantile", alpha=q, **self.kw)
            m.fit(train[self.feats_], train[TARGET])
            self.models_[q] = m
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return _reinflate(self.models_[0.5].predict(df[self.feats_]), df)

    def predict_interval(self, df: pd.DataFrame) -> pd.DataFrame:
        out = {f"p{int(q * 100)}": _reinflate(m.predict(df[self.feats_]), df)
               for q, m in self.models_.items()}
        # Quantile crossing is possible; enforce monotonicity.
        lo, mid, hi = out["p10"], out["p50"], out["p90"]
        lo, hi = np.minimum(lo, mid), np.maximum(hi, mid)
        return pd.DataFrame({"p10": lo, "p50": mid, "p90": hi}, index=df.index)


class Conformal:
    """Split-conformal intervals around any point model.

    Calibration residuals are taken in log space, so the euro interval is a
    multiplicative band around the prediction.

    Two caveats worth knowing. The `p10`/`p90` column names are a convenience:
    this is a symmetric fixed-width band at the nominal confidence level, not an
    estimate of the 10th and 90th percentiles, and the two coincide only if the
    residual distribution is symmetric. And conformal's finite-sample coverage
    guarantee assumes exchangeability between calibration and test, which a
    temporal split breaks - measured coverage here sits just under nominal
    (0.79 against 0.80) for exactly that reason.
    """

    def __init__(self, base, confidence: float = 0.8, calib_frac: float = 0.25,
                 random_state: int = 0):
        self.base = base
        self.confidence = confidence
        self.calib_frac = calib_frac
        self.random_state = random_state
        self.name = f"Conformal({base.name})"

    def fit(self, train: pd.DataFrame, valid: pd.DataFrame | None = None):
        # Calibrate on a player-disjoint slice so the residuals are honest.
        rng = np.random.default_rng(self.random_state)
        players = train.tm_url.dropna().unique()
        held = set(rng.choice(players, size=max(1, int(len(players) * self.calib_frac)),
                              replace=False))
        mask = train.tm_url.isin(held)
        fit_part, calib = train[~mask], train[mask]
        if len(calib) < 50:
            # Falling back to in-sample calibration would use residuals far
            # smaller than out-of-sample ones and produce intervals that look
            # tight and cover nothing. Better to fail loudly.
            raise ValueError(
                f"conformal calibration slice too small ({len(calib)} rows from "
                f"{len(train)}); lower calib_frac or supply more training data")

        self.base.fit(fit_part, valid)
        pred = np.log1p(np.clip(self.base.predict(calib), 1e4, None))
        resid = np.abs(np.log1p(calib.value_eur.to_numpy()) - pred)
        n = len(resid)
        # Finite-sample corrected conformal quantile.
        level = min(1.0, np.ceil((n + 1) * self.confidence) / n)
        self.q_ = float(np.quantile(resid, level))
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return self.base.predict(df)

    def predict_interval(self, df: pd.DataFrame) -> pd.DataFrame:
        mid = np.clip(self.predict(df), 1e4, None)
        lmid = np.log1p(mid)
        return pd.DataFrame({
            "p10": np.expm1(lmid - self.q_),
            "p50": mid,
            "p90": np.expm1(lmid + self.q_),
        }, index=df.index)


def interval_report(iv: pd.DataFrame, y_true: np.ndarray) -> dict[str, float]:
    """Coverage (PICP) and how wide the band has to be to get it."""
    lo, hi = iv.p10.to_numpy(), iv.p90.to_numpy()
    covered = (y_true >= lo) & (y_true <= hi)
    return {
        "picp": float(covered.mean()),
        "median_width_m": float(np.median(hi - lo) / 1e6),
        "median_width_ratio": float(np.median(hi / np.maximum(lo, 1e4))),
    }
