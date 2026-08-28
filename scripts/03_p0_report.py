"""P0: correctness spot-check and the signal figure."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
df = pd.read_parquet(ROOT / "data/processed/p0_panel.parquet")
d = df.dropna(subset=["value_eur", "npxg_xag_per90", "age_exact"])

# ---- correctness spot-check -------------------------------------------------
print("=== spot-check: highest-valued links, 2021-22 season stats -> 2022 value ===")
s = d[d.Season_End_Year == 2022].nlargest(12, "value_eur")
print(s[["Player", "Squad", "age_exact", "Min_Playing", "npxg_xag_per90",
         "value_eur", "tier"]].assign(
    age_exact=lambda x: x.age_exact.round(1),
    value_m=lambda x: (x.value_eur / 1e6).round(0),
).drop(columns="value_eur").to_string(index=False))

print("\n=== name-variant cases that Tier C/D caught ===")
hard = d[d.tier.isin(["C", "D"])].nlargest(8, "value_eur")
print(hard[["Player", "Squad", "tier", "value_eur"]].assign(
    value_m=lambda x: (x.value_eur / 1e6).round(1)).drop(columns="value_eur").to_string(index=False))

# ---- figure -----------------------------------------------------------------
fig, ax = plt.subplots(1, 3, figsize=(15.5, 4.6))
fig.suptitle("P0 kill-risk spike — Premier League, 2017-18 to 2021-22 "
             f"(n={len(d):,} player-seasons, 99.1% FBref↔Transfermarkt match)",
             fontsize=11, y=1.02)

sc = ax[0].scatter(d.npxg_xag_per90, d.value_eur / 1e6, c=d.age_exact,
                   cmap="viridis_r", s=13, alpha=.65, linewidths=0)
ax[0].set_yscale("log")
ax[0].set_xlabel("npxG + xAG per 90")
ax[0].set_ylabel("Transfermarkt value (€m, log scale)")
ax[0].set_title("Output vs value", fontsize=10)
plt.colorbar(sc, ax=ax[0], label="age")

bins = np.arange(17, 39, 1)
d["_ab"] = pd.cut(d.age_exact, bins)
med = d.groupby("_ab", observed=True).value_eur.median() / 1e6
ctr = [i.mid for i in med.index]
ax[1].scatter(d.age_exact, d.value_eur / 1e6, s=10, alpha=.28, color="#4A5666", linewidths=0)
ax[1].plot(ctr, med.values, color="#C1121F", lw=2.4, label="median by age")
ax[1].set_yscale("log")
ax[1].set_xlabel("age at valuation")
ax[1].set_ylabel("value (€m, log)")
ax[1].set_title("The age curve", fontsize=10)
ax[1].legend(fontsize=8, frameon=False)

by = d.groupby("Season_End_Year").value_eur.median() / 1e6
ax[2].bar([f"{y-1}-{str(y)[2:]}" for y in by.index], by.values, color="#2A6F97")
ax[2].set_ylabel("median value (€m)")
ax[2].set_title("Market inflation → must be deflated", fontsize=10)
ax[2].tick_params(axis="x", rotation=30, labelsize=8)

for a in ax:
    a.spines[["top", "right"]].set_visible(False)
    a.grid(alpha=.25, lw=.6)
plt.tight_layout()
out = ROOT / "data/processed/p0_signal.png"
plt.savefig(out, dpi=155, bbox_inches="tight")
print(f"\nfigure -> {out}")
