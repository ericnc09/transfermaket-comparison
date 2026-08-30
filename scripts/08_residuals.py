"""P4: where the model disagrees with Transfermarkt, and why.

Residuals come from the `coldstart` model - the one that has never seen a
Transfermarkt value. Disagreement from the `update` model would be far less
interesting, since it is handed the previous valuation as a feature.
"""
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
V = "coldstart"
MIN_TM = 3_000_000        # ignore the noisy bottom of the market
N_STRATA = 10


def main() -> None:
    p = pd.read_parquet(PROC / "predictions.parquet")
    p["log_resid"] = np.log1p(p[f"pred_{V}"]) - np.log1p(p.value_eur)

    # Any regression shrinks toward the mean, so raw residuals run from +0.36 in
    # the cheapest decile to -0.34 in the dearest. Ranking on them would simply
    # return cheap players as "undervalued" and expensive ones as "overvalued".
    # Comparing each player against others Transfermarkt prices similarly is the
    # only way the ranking says anything about the player rather than the price.
    p["stratum"] = pd.qcut(p.value_eur, N_STRATA, labels=False, duplicates="drop")
    p["resid_adj"] = p.log_resid - p.groupby("stratum").log_resid.transform("mean")

    big = p[p.value_eur >= MIN_TM].copy()

    cols = ["Player", "Comp", "Season_End_Year", "pos_group", "age",
            "minutes", "value_eur", f"pred_{V}", f"ratio_{V}"]

    def show(df):
        d = df[cols].copy()
        d["tm_m"] = (d.value_eur / 1e6).round(1)
        d["model_m"] = (d[f"pred_{V}"] / 1e6).round(1)
        d["ratio"] = d[f"ratio_{V}"].round(2)
        d["age"] = d.age.round(1)
        d["season"] = d.Season_End_Year.astype(int)
        return d[["Player", "Comp", "season", "pos_group", "age",
                  "tm_m", "model_m", "ratio"]].to_string(index=False)

    print(f"=== MODEL SAYS UNDERVALUED (top 15, TM >= EUR{MIN_TM/1e6:.0f}m) ===")
    print(show(big.nlargest(15, "resid_adj")))
    print(f"\n=== MODEL SAYS OVERVALUED (top 15) ===")
    print(show(big.nsmallest(15, "resid_adj")))

    print("\n=== where the model disagrees, after adjusting for price level ===")
    print("(raw = uncorrected, dominated by regression to the mean;"
          " price_adjusted is the meaningful column)")
    for by, bins in [("pos_group", None), ("Comp", None), ("age", 5)]:
        g = pd.qcut(big[by], bins, duplicates="drop") if bins else big[by]
        t = big.groupby(g, observed=True).agg(
            n=("resid_adj", "size"),
            raw=("log_resid", "mean"),
            price_adjusted=("resid_adj", "mean"))
        print(f"\nby {by}:")
        print(t.round(3).to_string())

    # ---- figure ------------------------------------------------------------
    fig, ax = plt.subplots(1, 3, figsize=(15.5, 4.6))
    fig.suptitle("Model vs Transfermarkt — cold-start model, out-of-sample "
                 f"(n={len(big):,} player-seasons)", fontsize=11, y=1.02)

    ax[0].scatter(big.value_eur / 1e6, big[f"pred_{V}"] / 1e6, s=9, alpha=.3,
                  color="#2A6F97", linewidths=0)
    lim = [1, 250]
    ax[0].plot(lim, lim, color="#C1121F", lw=1.4, ls="--", label="perfect agreement")
    ax[0].set_xscale("log"); ax[0].set_yscale("log")
    ax[0].set_xlabel("Transfermarkt value (€m)")
    ax[0].set_ylabel("model value (€m)")
    ax[0].set_title("Agreement", fontsize=10); ax[0].legend(fontsize=8, frameon=False)

    ax[1].scatter(big.age, big.resid_adj, s=9, alpha=.28, color="#4A5666", linewidths=0)
    bins = np.arange(17, 39, 1)
    med = big.groupby(pd.cut(big.age, bins), observed=True).resid_adj.median()
    ax[1].plot([i.mid for i in med.index], med.values, color="#C1121F", lw=2.4)
    ax[1].axhline(0, color="#888", lw=1)
    ax[1].set_xlabel("age"); ax[1].set_ylabel("price-adjusted residual")
    ax[1].set_title("Disagreement by age, price-adjusted", fontsize=10)

    order = big.groupby("pos_group").resid_adj.median().sort_values()
    ax[2].barh(order.index, order.values, color="#2A6F97")
    ax[2].axvline(0, color="#888", lw=1)
    ax[2].set_xlabel("median price-adjusted residual")
    ax[2].set_title("Disagreement by position", fontsize=10)

    for a in ax:
        a.spines[["top", "right"]].set_visible(False)
        a.grid(alpha=.25, lw=.6)
    plt.tight_layout()
    out = PROC / "p4_residuals.png"
    plt.savefig(out, dpi=155, bbox_inches="tight")

    big.nlargest(200, "resid_adj").to_parquet(PROC / "p4_undervalued.parquet", index=False)
    big.nsmallest(200, "resid_adj").to_parquet(PROC / "p4_overvalued.parquet", index=False)
    print(f"\nfigure -> {out}")


if __name__ == "__main__":
    main()
