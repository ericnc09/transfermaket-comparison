"""P2: train the baselines and the first real models, and print the leaderboard."""
import json
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")

import pandas as pd

from src.eval.metrics import score, segment_scores
from src.eval.splits import TEST, TRAIN, VALID, temporal_split
from src.models import baselines as B
from src.models.gbm import CatBoost, HistGBM

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"


def main() -> None:
    df = pd.read_parquet(PROC / "panel_model.parquet")
    train, valid, test = temporal_split(df)
    print(f"train {len(train):,} ({TRAIN})   valid {len(valid):,} ({VALID})   "
          f"test {len(test):,} ({TEST})\n")

    models = [c() for c in B.ALL]
    models += [HistGBM("coldstart"), HistGBM("update"),
               CatBoost("coldstart"), CatBoost("update")]

    rows, preds = [], {}
    for m in models:
        m.fit(train, valid)
        p = m.predict(test)
        preds[m.name] = p
        rows.append({"model": m.name, **score(test.value_eur.to_numpy(), p)})

    lb = pd.DataFrame(rows).sort_values("log_mae")
    show = ["model", "log_mae", "log_r2", "mae_m", "medae_m",
            "within_30pct", "spearman", "ndcg_100"]
    print("=== TEST LEADERBOARD (2021-22 season, held out) ===")
    print(lb[show].to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    best = lb.iloc[0].model
    print(f"\n=== segments for: {best} ===")
    y, p = test.value_eur.to_numpy(), preds[best]
    for by, bins in [("pos_group", None), ("Comp", None),
                     ("age", 4), ("value_eur", 4)]:
        seg = segment_scores(test, y, p, by, bins)
        if not seg.empty:
            print(f"\nby {by}:")
            print(seg.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    lb.to_json(PROC / "p2_leaderboard.json", orient="records", indent=2)
    out = test[["Player", "Comp", "pos_group", "age", "minutes",
                "value_eur", "prior_value_eur"]].copy()
    for name, p in preds.items():
        out[name] = p
    out.to_parquet(PROC / "p2_test_predictions.parquet", index=False)
    print(f"\n-> {PROC/'p2_leaderboard.json'}")


if __name__ == "__main__":
    main()
