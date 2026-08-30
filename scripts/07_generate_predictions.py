"""P4: out-of-sample predictions and intervals for every row in the panel.

Each season is predicted by a model trained only on earlier seasons, so every
residual in the leaderboard is genuinely out-of-sample. Both variants are run:
`coldstart` drives the disagreement analysis, `update` is the accuracy view.

LightGBM is used for both. CatBoost edges it (0.401 vs 0.408 cold-start) but
that gap is a tenth of the fold-to-fold spread, and LightGBM explains cleanly
through SHAP without the categorical-Pool indirection.
"""
import json
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")

import joblib
import pandas as pd

from src.eval.metrics import score
from src.eval.splits import rolling_origin
from src.models.gbm import LightGBM
from src.models.quantile import Conformal, interval_report

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
MODELS = PROC / "models"
VARIANTS = ("coldstart", "update")


def main() -> None:
    MODELS.mkdir(parents=True, exist_ok=True)
    panel = pd.read_parquet(PROC / "panel_model.parquet")
    params = {}
    pfile = PROC / "p3_best_params.json"
    if pfile.exists():
        params = json.loads(pfile.read_text())

    frames, last = [], None
    for season, train, test in rolling_origin(panel):
        block = test.copy()
        for variant in VARIANTS:
            kw = params.get(f"LightGBM|{variant}", {})
            model = Conformal(LightGBM(variant, **kw), confidence=0.8).fit(train)
            iv = model.predict_interval(test)
            block[f"pred_{variant}"] = iv.p50.to_numpy()
            block[f"lo_{variant}"] = iv.p10.to_numpy()
            block[f"hi_{variant}"] = iv.p90.to_numpy()
            if season == max(s for s, _, _ in rolling_origin(panel)):
                joblib.dump(model, MODELS / f"conformal_lgbm_{variant}.joblib")
        frames.append(block)
        last = (season, train, test)
        print(f"fold {season}: {len(test):,} rows predicted", flush=True)

    out = pd.concat(frames, ignore_index=True)
    for variant in VARIANTS:
        out[f"delta_{variant}"] = out[f"pred_{variant}"] - out.value_eur
        out[f"ratio_{variant}"] = out[f"pred_{variant}"] / out.value_eur

    out.to_parquet(PROC / "predictions.parquet", index=False)
    joblib.dump(last[1], MODELS / "background.joblib")

    print(f"\n{len(out):,} out-of-sample predictions across "
          f"{out.Season_End_Year.nunique()} seasons")
    y = out.value_eur.to_numpy()
    for variant in VARIANTS:
        s = score(y, out[f"pred_{variant}"].to_numpy())
        cov = ((y >= out[f"lo_{variant}"]) & (y <= out[f"hi_{variant}"])).mean()
        print(f"  {variant:<10} log_mae={s['log_mae']:.3f}  "
              f"medae=EUR{s['medae_m']:.2f}m  within30%={s['within_30pct']:.3f}  "
              f"interval coverage={cov:.3f}")
    print(f"\n-> {PROC/'predictions.parquet'}")


if __name__ == "__main__":
    main()
