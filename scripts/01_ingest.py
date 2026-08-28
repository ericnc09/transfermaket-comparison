"""P0 ingest: pull every source table into data/raw/ as parquet."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.ingest.sources import SOURCES, load

if __name__ == "__main__":
    for name in SOURCES:
        df = load(name)
        print(f"{name:<22} {df.shape[0]:>7,} rows x {df.shape[1]:>3} cols")
