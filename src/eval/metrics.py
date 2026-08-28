"""Scoring. Every model predicts euros; metrics are computed uniformly on that."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as sps

EPS = 1e-9


def ndcg_at_k(y_true: np.ndarray, y_pred: np.ndarray, k: int = 100) -> float:
    """Do the expensive players come out on top? Gain is the true value."""
    order = np.argsort(-y_pred)[:k]
    disc = 1.0 / np.log2(np.arange(2, len(order) + 2))
    dcg = float((y_true[order] * disc).sum())
    ideal = np.sort(y_true)[::-1][:k]
    idcg = float((ideal * disc[: len(ideal)]).sum())
    return dcg / idcg if idcg > 0 else np.nan


def score(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, float)
    y_pred = np.clip(np.asarray(y_pred, float), 1e4, None)
    lt, lp = np.log1p(y_true), np.log1p(y_pred)
    resid = lt - lp

    ratio = y_pred / np.maximum(y_true, EPS)
    smape = np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_pred) + np.abs(y_true) + EPS))

    return {
        "log_rmse": float(np.sqrt(np.mean(resid ** 2))),
        "log_mae": float(np.mean(np.abs(resid))),
        "log_r2": float(1 - np.sum(resid ** 2) / np.sum((lt - lt.mean()) ** 2)),
        "mae_m": float(np.mean(np.abs(y_pred - y_true)) / 1e6),
        "medae_m": float(np.median(np.abs(y_pred - y_true)) / 1e6),
        "smape": float(smape),
        "within_30pct": float(np.mean((ratio > 0.7) & (ratio < 1.3))),
        "spearman": float(sps.spearmanr(y_true, y_pred).statistic),
        "ndcg_100": ndcg_at_k(y_true, y_pred, 100),
    }


def segment_scores(df: pd.DataFrame, y_true: np.ndarray, y_pred: np.ndarray,
                   by: str, bins=None) -> pd.DataFrame:
    """Break a metric set out by position, league, age band or value decile."""
    g = pd.qcut(df[by], bins, duplicates="drop") if bins else df[by]
    rows = []
    for key, idx in pd.Series(range(len(df))).groupby(g.values, observed=True):
        i = idx.values
        if len(i) < 25:
            continue
        rows.append({by: str(key), "n": len(i),
                     **{k: v for k, v in score(y_true[i], y_pred[i]).items()
                        if k in ("log_mae", "medae_m", "within_30pct", "spearman")}})
    return pd.DataFrame(rows)
