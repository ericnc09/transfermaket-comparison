"""P1: build the modelling panel and persist it to parquet + DuckDB."""
import json
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore", category=FutureWarning)

import duckdb
import pandas as pd

from src.features.build import build_panel
from src.features.definitions import DISCONTINUED_2023
from src.features.manifest import CATEGORICAL, feature_columns

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"


def main() -> None:
    PROC.mkdir(parents=True, exist_ok=True)
    df = build_panel()
    df.to_parquet(PROC / "panel_full.parquet", index=False)

    model = df[df.eligible & df.value_eur.notna()].copy()
    model.to_parquet(PROC / "panel_model.parquet", index=False)

    con = duckdb.connect(str(PROC / "market.duckdb"))
    con.execute("CREATE OR REPLACE TABLE panel_full AS SELECT * FROM df")
    con.execute("CREATE OR REPLACE TABLE panel_model AS SELECT * FROM model")
    con.close()

    cold = feature_columns(model, "coldstart")
    upd = feature_columns(model, "update")
    manifest = {
        "rows_full": int(len(df)),
        "rows_model": int(len(model)),
        "unique_players": int(model.tm_url.nunique()),
        "seasons": sorted(int(s) for s in model.Season_End_Year.unique()),
        "n_features_coldstart": len(cold),
        "n_features_update": len(upd),
        "features_coldstart": cold,
        "features_update": upd,
        "categorical": CATEGORICAL,
        "discontinued_2023": DISCONTINUED_2023,
        "target": "log_value",
        "target_deflated": "log_value_deflated",
    }
    (PROC / "panel_manifest.json").write_text(json.dumps(manifest, indent=2))

    print(f"panel_full   {len(df):>6,} rows x {df.shape[1]} cols")
    print(f"panel_model  {len(model):>6,} rows  ({model.tm_url.nunique():,} unique players)")
    print(f"features     coldstart={len(cold)}  update={len(upd)}")
    print(f"\nrows by season:")
    print(model.groupby("Season_End_Year").size().to_string())
    print(f"\nrows by league:")
    print(model.groupby("Comp").size().to_string())
    print(f"\n-> {PROC/'market.duckdb'}")


if __name__ == "__main__":
    main()
