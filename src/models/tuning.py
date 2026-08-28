"""Optuna tuning with player-grouped CV.

Folds are grouped on player so the same footballer never sits in both sides of a
split. Ungrouped CV would let a player's 2018-19 row tune against his 2019-20
row, which flatters every model and flatters the high-variance ones most.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.eval.metrics import score
from src.eval.splits import grouped_folds

# Per-family trial budgets. Cost per fit differs by more than an order of
# magnitude, so a flat budget would spend most of the wall clock on EBM - which
# is the slowest family and not the accuracy leader. Spend where it buys most.
TRIAL_BUDGET = {
    "LightGBM": 1.0,
    "XGBoost": 1.0,
    "HistGBM": 1.0,
    "CatBoost": 0.6,
    "EBM": 0.3,
}

SPACES = {
    "LightGBM": lambda t: dict(
        n_estimators=t.suggest_int("n_estimators", 300, 2500, step=100),
        learning_rate=t.suggest_float("learning_rate", 0.01, 0.15, log=True),
        num_leaves=t.suggest_int("num_leaves", 15, 127, log=True),
        min_child_samples=t.suggest_int("min_child_samples", 5, 80),
        subsample=t.suggest_float("subsample", 0.6, 1.0),
        colsample_bytree=t.suggest_float("colsample_bytree", 0.4, 1.0),
        reg_lambda=t.suggest_float("reg_lambda", 1e-3, 30.0, log=True),
    ),
    "XGBoost": lambda t: dict(
        n_estimators=t.suggest_int("n_estimators", 300, 2500, step=100),
        learning_rate=t.suggest_float("learning_rate", 0.01, 0.15, log=True),
        max_depth=t.suggest_int("max_depth", 3, 10),
        min_child_weight=t.suggest_float("min_child_weight", 1.0, 30.0, log=True),
        subsample=t.suggest_float("subsample", 0.6, 1.0),
        colsample_bytree=t.suggest_float("colsample_bytree", 0.4, 1.0),
        reg_lambda=t.suggest_float("reg_lambda", 1e-3, 30.0, log=True),
    ),
    "CatBoost": lambda t: dict(
        iterations=t.suggest_int("iterations", 400, 3000, step=100),
        learning_rate=t.suggest_float("learning_rate", 0.01, 0.15, log=True),
        depth=t.suggest_int("depth", 4, 9),
        l2_leaf_reg=t.suggest_float("l2_leaf_reg", 0.5, 30.0, log=True),
    ),
    "HistGBM": lambda t: dict(
        max_iter=t.suggest_int("max_iter", 200, 1500, step=100),
        learning_rate=t.suggest_float("learning_rate", 0.01, 0.15, log=True),
        max_leaf_nodes=t.suggest_int("max_leaf_nodes", 15, 127, log=True),
        min_samples_leaf=t.suggest_int("min_samples_leaf", 5, 80),
        l2_regularization=t.suggest_float("l2_regularization", 1e-3, 30.0, log=True),
    ),
    "EBM": lambda t: dict(
        learning_rate=t.suggest_float("learning_rate", 0.005, 0.08, log=True),
        interactions=t.suggest_int("interactions", 0, 20),
        outer_bags=t.suggest_int("outer_bags", 4, 14),
    ),
}


def cv_log_mae(model_cls, params: dict, train: pd.DataFrame,
               variant: str, n_splits: int = 4) -> float:
    """Mean log MAE over player-grouped folds of the training window."""
    losses = []
    for tr_i, va_i in grouped_folds(train, n_splits):
        tr, va = train.iloc[tr_i], train.iloc[va_i]
        m = model_cls(variant, **params).fit(tr, None)
        losses.append(score(va.value_eur.to_numpy(), m.predict(va))["log_mae"])
    return float(np.mean(losses))


def tune(model_cls, family: str, train: pd.DataFrame, variant: str = "coldstart",
         n_trials: int = 40, n_splits: int = 4, seed: int = 0) -> dict:
    """Return the best hyperparameters for one family/variant."""
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    space = SPACES[family]

    def objective(trial):
        return cv_log_mae(model_cls, space(trial), train, variant, n_splits)

    study = optuna.create_study(
        direction="minimize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return {"best_params": study.best_params,
            "best_cv_log_mae": float(study.best_value),
            "n_trials": n_trials}
