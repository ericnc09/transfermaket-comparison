"""P2/P3: evaluate the model zoo with rolling-origin cross-validation.

A single train/valid/test split spends most of a four-season panel on a
two-season training window and reports one number from one season. Rolling
origin tests on every season in turn, giving a mean and a spread.
"""
import json
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from src.eval.metrics import score, segment_scores
from src.eval.splits import rolling_origin, temporal_split
from src.models import baselines as B
from src.models.gbm import CatBoost, HistGBM, LightGBM, XGBoost

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
GBMS = [HistGBM, CatBoost, LightGBM, XGBoost]
SHOW = ["log_mae", "log_r2", "mae_m", "medae_m", "within_30pct", "spearman", "ndcg_100"]


def build_models():
    models = [c() for c in B.ALL]
    for cls in GBMS:
        for variant in ("coldstart", "update"):
            models.append(cls(variant))
    return models


def _inner_valid(train: pd.DataFrame):
    """Hold out the most recent training season for early stopping."""
    seasons = sorted(train.Season_End_Year.unique())
    if len(seasons) < 2:
        return train, None
    return train[train.Season_End_Year < seasons[-1]], train[train.Season_End_Year == seasons[-1]]


def main() -> None:
    df = pd.read_parquet(PROC / "panel_model.parquet")
    print(f"panel {len(df):,} rows · {df.tm_url.nunique():,} players · "
          f"seasons {sorted(int(s) for s in df.Season_End_Year.unique())}\n")

    records, fold_preds = [], {}
    for season, train, test in rolling_origin(df):
        fit_tr, fit_va = _inner_valid(train)
        for m in build_models():
            m.fit(fit_tr, fit_va)
            p = m.predict(test)
            records.append({"model": m.name, "fold": season, "n_test": len(test),
                            **score(test.value_eur.to_numpy(), p)})
            fold_preds[(m.name, season)] = (test, p)

    r = pd.DataFrame(records)
    agg = (r.groupby("model")[SHOW].agg(["mean", "std"]))
    agg.columns = [f"{a}_{b}" for a, b in agg.columns]
    agg = agg.sort_values("log_mae_mean")

    print("=== ROLLING-ORIGIN LEADERBOARD (mean over 3 test seasons) ===")
    out = agg[["log_mae_mean", "log_mae_std", "log_r2_mean", "mae_m_mean",
               "medae_m_mean", "within_30pct_mean", "spearman_mean", "ndcg_100_mean"]]
    out.columns = ["log_mae", "±sd", "log_r2", "mae_m", "medae_m", "±30%", "rho", "ndcg"]
    print(out.to_string(float_format=lambda x: f"{x:.3f}"))

    print("\n=== per-fold log MAE ===")
    piv = r.pivot_table(index="model", columns="fold", values="log_mae")
    print(piv.reindex(agg.index).to_string(float_format=lambda x: f"{x:.3f}"))

    best = agg.index[0]
    last = max(r.fold)
    test, p = fold_preds[(best, last)]
    print(f"\n=== segments · {best} · test {last-1}-{str(last)[2:]} ===")
    y = test.value_eur.to_numpy()
    for by, bins in [("pos_group", None), ("Comp", None), ("age", 4), ("value_eur", 4)]:
        seg = segment_scores(test, y, p, by, bins)
        if not seg.empty:
            print(f"\nby {by}:")
            print(seg.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    r.to_json(PROC / "p2_folds.json", orient="records", indent=2)
    agg.to_json(PROC / "p2_leaderboard.json", indent=2)

    # Hold-out predictions from the final fold, for the residual work in P4.
    keep = test[["Player", "Comp", "pos_group", "age", "minutes",
                 "value_eur", "prior_value_eur"]].copy()
    for (name, season), (t, pr) in fold_preds.items():
        if season == last:
            keep[name] = pr
    keep.to_parquet(PROC / "p2_test_predictions.parquet", index=False)
    print(f"\n-> {PROC/'p2_leaderboard.json'}")


if __name__ == "__main__":
    main()
