"""Regenerate the headline leaderboard with the current code.

Exists because the published leaderboard once went stale against changes to the
target and the feature set. Run this after touching gbm.py, manifest.py or the
feature builder, and before quoting a number anywhere.
"""
import sys, warnings; sys.path.insert(0,'.'); warnings.filterwarnings('ignore')
import json, numpy as np, pandas as pd
from src.eval.splits import rolling_origin
from src.eval.metrics import score
from src.models import baselines as B
from src.models.gbm import LightGBM, CatBoost
panel = pd.read_parquet('data/processed/panel_model.parquet')
params = json.load(open('data/processed/p3_best_params.json'))
rows=[]
for s, tr, te in rolling_origin(panel):
    y = te.value_eur.to_numpy()
    ms = [c() for c in B.ALL]
    for v in ("coldstart","update"):
        ms += [LightGBM(v, **params.get(f"LightGBM|{v}",{})),
               CatBoost(v, **params.get(f"CatBoost|{v}",{}))]
    for m in ms:
        m.fit(tr, None)
        rows.append({"model":m.name,"fold":s,**score(y,m.predict(te))})
    print(f"fold {s} done", flush=True)
r=pd.DataFrame(rows)
agg=r.groupby("model")[["log_mae","log_r2","medae_m","within_30pct","ndcg_100"]].mean()
agg["sd"]=r.groupby("model").log_mae.std()
print("\n=== CORRECTED LEADERBOARD (current code) ===")
print(agg.sort_values("log_mae").to_string(float_format=lambda x:f"{x:.3f}"))
agg.to_json('data/processed/p3_leaderboard_corrected.json', indent=2)
