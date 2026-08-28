"""P3: tune the model zoo, evaluate with rolling origin, calibrate intervals.

    python scripts/06_tune_and_evaluate.py --trials 40
    python scripts/06_tune_and_evaluate.py --trials 0     # skip tuning, defaults only

Tuning uses player-grouped CV inside the training window of the final fold, so
no hyperparameter is ever chosen using a season it is later scored on.
"""
import argparse
import json
import sys
import time
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from src.eval.metrics import score
from src.eval.splits import rolling_origin
from src.models import baselines as B
from src.models.ebm import EBM
from src.models.ensemble import Stacked
from src.models.gbm import CatBoost, HistGBM, LightGBM, XGBoost
from src.models.linear import ElasticNetSpline, RidgeSpline
from src.models.quantile import Conformal, QuantileTrio, interval_report
from src.models.tuning import tune

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"

FAMILIES = {"LightGBM": LightGBM, "CatBoost": CatBoost,
            "XGBoost": XGBoost, "HistGBM": HistGBM, "EBM": EBM}
SHOW = ["log_mae", "log_r2", "mae_m", "medae_m", "within_30pct", "spearman", "ndcg_100"]


def main(trials: int, variants: list[str]) -> None:
    df = pd.read_parquet(PROC / "panel_model.parquet")
    print(f"panel {len(df):,} rows · {df.tm_url.nunique():,} players · "
          f"seasons {sorted(int(s) for s in df.Season_End_Year.unique())}", flush=True)

    folds = list(rolling_origin(df))
    print(f"{len(folds)} rolling-origin folds\n", flush=True)

    # ---- tuning, on the last fold's training window only -------------------
    best: dict[tuple[str, str], dict] = {}
    if trials:
        _, tune_train, _ = folds[-1]
        for fam, cls in FAMILIES.items():
            for variant in variants:
                t0 = time.time()
                res = tune(cls, fam, tune_train, variant, n_trials=trials)
                best[(fam, variant)] = res["best_params"]
                print(f"tuned {fam:<9} [{variant:<9}] "
                      f"cv_log_mae={res['best_cv_log_mae']:.4f} "
                      f"({time.time()-t0:.0f}s)", flush=True)
        (PROC / "p3_best_params.json").write_text(json.dumps(
            {f"{f}|{v}": p for (f, v), p in best.items()}, indent=2))
        print(flush=True)

    def build():
        models = [c() for c in B.ALL]
        for variant in variants:
            models += [RidgeSpline(variant), ElasticNetSpline(variant)]
            for fam, cls in FAMILIES.items():
                models.append(cls(variant, **best.get((fam, variant), {})))
            models.append(Stacked(
                [LightGBM(variant, **best.get(("LightGBM", variant), {})),
                 CatBoost(variant, **best.get(("CatBoost", variant), {})),
                 EBM(variant, **best.get(("EBM", variant), {}))],
                name=f"Stacked [{variant}]"))
        return models

    # ---- rolling-origin evaluation -----------------------------------------
    records = []
    for season, train, test in folds:
        y = test.value_eur.to_numpy()
        for m in build():
            m.fit(train, None)
            records.append({"model": m.name, "fold": season,
                            **score(y, m.predict(test))})
        print(f"fold {season} done ({len(test):,} test rows)", flush=True)

    r = pd.DataFrame(records)
    agg = r.groupby("model")[SHOW].mean().sort_values("log_mae")
    agg["log_mae_sd"] = r.groupby("model").log_mae.std()
    print("\n=== ROLLING-ORIGIN LEADERBOARD ===")
    print(agg.to_string(float_format=lambda x: f"{x:.3f}"))

    # ---- interval calibration ----------------------------------------------
    print("\n=== INTERVAL CALIBRATION (final fold) ===")
    _, tr, te = folds[-1]
    y = te.value_eur.to_numpy()
    iv_rows = []
    for variant in variants:
        q = QuantileTrio(variant).fit(tr)
        iv_rows.append({"method": f"LGBM quantile [{variant}]", **interval_report(q.predict_interval(te), y)})
        c = Conformal(LightGBM(variant, **best.get(("LightGBM", variant), {})),
                      confidence=0.8).fit(tr)
        iv_rows.append({"method": f"conformal 80% [{variant}]", **interval_report(c.predict_interval(te), y)})
    ivdf = pd.DataFrame(iv_rows)
    print(ivdf.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print("\n(picp is realised coverage; nominal is 0.80)")

    r.to_json(PROC / "p3_folds.json", orient="records", indent=2)
    agg.to_json(PROC / "p3_leaderboard.json", indent=2)
    ivdf.to_json(PROC / "p3_intervals.json", orient="records", indent=2)
    print(f"\n-> {PROC/'p3_leaderboard.json'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=30)
    ap.add_argument("--variants", nargs="+", default=["coldstart", "update"])
    a = ap.parse_args()
    main(a.trials, a.variants)
