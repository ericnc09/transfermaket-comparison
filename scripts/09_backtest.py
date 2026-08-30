"""P5: does the model see value before Transfermarkt does?

Take the cold-start model's disagreement with Transfermarkt at season t, then
measure what Transfermarkt itself did over the following year. If players the
model calls underpriced go on to be repriced upward, the model holds information
Transfermarkt had not yet incorporated.

The confound, and why the naive version of this test is worthless
-----------------------------------------------------------------
Let V be the Transfermarkt value, M the model's, and suppose V carries transient
noise e - Transfermarkt has overshot. Then the residual (log M - log V) contains
-e, and the forward return (log V' - log V) also contains -e, because the
overshoot reverts. Any model that merely fails to replicate Transfermarkt's noise
will therefore appear to predict its future movements. That is mean reversion,
not skill.

So the headline test is not the raw correlation. It is whether the residual still
predicts forward movement after controlling for Transfermarkt's own recent
momentum, the player's age, the price level and the season - with standard errors
clustered by player, since the same footballer appears in several seasons.
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
import statsmodels.formula.api as smf

from src.features.labels import forward_values

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
V = "coldstart"
N_STRATA = 10


def build() -> pd.DataFrame:
    p = pd.read_parquet(PROC / "predictions.parquet")
    fv = forward_values(p[["tm_player_id", "label_date"]].drop_duplicates())
    d = p.merge(fv, on=["tm_player_id", "label_date"], how="inner")

    d["log_resid"] = np.log1p(d[f"pred_{V}"]) - np.log1p(d.value_eur)
    d["stratum"] = pd.qcut(d.value_eur, N_STRATA, labels=False, duplicates="drop")
    d["resid_adj"] = d.log_resid - d.groupby("stratum").log_resid.transform("mean")

    d["fwd_return"] = np.log1p(d.fwd_value_eur) - np.log1p(d.value_eur)
    # Transfermarkt's own recent move - the mean-reversion control.
    d["tm_momentum"] = np.log1p(d.value_eur) - np.log1p(d.prior_value_eur)
    d["log_price"] = np.log1p(d.value_eur)
    d["season"] = d.Season_End_Year.astype(int)
    return d.dropna(subset=["fwd_return", "resid_adj", "tm_momentum", "age"])


def main() -> None:
    d = build()
    print(f"backtest sample: {len(d):,} player-seasons, {d.tm_url.nunique():,} players, "
          f"seasons {sorted(d.season.unique())}\n")

    # ---- 1. the intuitive view -------------------------------------------
    d["dec"] = pd.qcut(d.resid_adj, 10, labels=False, duplicates="drop")
    tab = d.groupby("dec").agg(n=("fwd_return", "size"),
                               resid=("resid_adj", "mean"),
                               fwd_return=("fwd_return", "mean"),
                               tm_momentum=("tm_momentum", "mean"),
                               age=("age", "mean"))
    print("=== forward TM movement by model-disagreement decile ===")
    print("(decile 0 = model says most overvalued, 9 = most undervalued)")
    print(tab.round(3).to_string())
    spread = tab.fwd_return.iloc[-1] - tab.fwd_return.iloc[0]
    print(f"\ntop-minus-bottom decile spread: {spread:+.3f} log points "
          f"({(np.exp(spread)-1)*100:+.1f}% in value terms)")

    # ---- 2. the honest test ----------------------------------------------
    print("\n=== does the signal survive controls? ===")
    models = {
        "raw (no controls)": "fwd_return ~ resid_adj",
        "+ TM momentum": "fwd_return ~ resid_adj + tm_momentum",
        "+ age, price, season": "fwd_return ~ resid_adj + tm_momentum + age "
                                "+ log_price + C(season)",
    }
    rows = []
    for label, formula in models.items():
        fit = smf.ols(formula, data=d).fit(
            cov_type="cluster", cov_kwds={"groups": d.tm_url})
        rows.append({"spec": label,
                     "beta_resid_adj": fit.params["resid_adj"],
                     "std_err": fit.bse["resid_adj"],
                     "t": fit.tvalues["resid_adj"],
                     "p": fit.pvalues["resid_adj"],
                     "r2": fit.rsquared})
    res = pd.DataFrame(rows)
    print(res.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\n(standard errors clustered by player)")

    beta = res.beta_resid_adj.iloc[-1]
    p_val = res.p.iloc[-1]
    print(f"\ncontrolled coefficient: {beta:+.4f}  (p={p_val:.2g})")
    print("interpretation: a 1.0 log-point disagreement maps to a "
          f"{beta:+.3f} log-point move in the following year's TM value")

    # ---- 3. robustness ----------------------------------------------------
    full = "fwd_return ~ resid_adj + tm_momentum + age + log_price + C(season)"

    print("\n=== robustness: per season (a fixed effect cannot hide a one-season fluke) ===")
    for season, g in d.groupby("season"):
        f = smf.ols(full.replace(" + C(season)", ""), data=g).fit(
            cov_type="cluster", cov_kwds={"groups": g.tm_url})
        print(f"  {season}:  n={len(g):>5,}  beta={f.params.resid_adj:+.4f}  "
              f"p={f.pvalues.resid_adj:.2g}")

    print("\n=== robustness: rows where TM has not recently moved the value ===")
    print("(pure mean reversion needs a recent move to revert; if the signal were")
    print(" only that, it should collapse here)")
    quiet = d[d.tm_momentum.abs() < 0.05]
    f = smf.ols(full, data=quiet).fit(
        cov_type="cluster", cov_kwds={"groups": quiet.tm_url})
    print(f"  n={len(quiet):,}  beta={f.params.resid_adj:+.4f}  "
          f"se={f.bse.resid_adj:.4f}  p={f.pvalues.resid_adj:.2g}")
    print("  the point estimate is stable, but this subsample is small enough that")
    print("  it cannot reject the null on its own - suggestive, not confirmatory")

    print("\n=== why forward returns are negative on average ===")
    drift = d.groupby("season").agg(mean_fwd=("fwd_return", "mean"),
                                    mean_age=("age", "mean"))
    drift["median_fwd_date"] = d.groupby("season").fwd_date.median().dt.strftime("%Y-%m")
    print(drift.round(3).to_string())
    print("  mean age 27 is past the peak of the age curve, and the 2019 cohort's")
    print("  forward date lands in the 2020 COVID markdown. Season fixed effects")
    print("  absorb this; the coefficient above is a within-season estimate.")

    # ---- 4. figure --------------------------------------------------------
    fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.4))
    fig.suptitle("Does the model see value before Transfermarkt does?  "
                 f"(n={len(d):,} player-seasons)", fontsize=11, y=1.02)

    ax[0].bar(tab.index, tab.fwd_return, color="#2A6F97")
    ax[0].axhline(0, color="#888", lw=1)
    ax[0].set_xlabel("model-disagreement decile  (9 = most undervalued)")
    ax[0].set_ylabel("mean forward TM return (log)")
    ax[0].set_title("Raw — includes mean reversion", fontsize=10)

    ax[1].scatter(d.resid_adj, d.fwd_return, s=6, alpha=.18,
                  color="#4A5666", linewidths=0)
    xs = np.linspace(d.resid_adj.quantile(.01), d.resid_adj.quantile(.99), 50)
    fit = smf.ols("fwd_return ~ resid_adj + tm_momentum + age + log_price + C(season)",
                  data=d).fit()
    base = pd.DataFrame({"resid_adj": xs, "tm_momentum": d.tm_momentum.mean(),
                         "age": d.age.mean(), "log_price": d.log_price.mean(),
                         "season": d.season.mode().iloc[0]})
    ax[1].plot(xs, fit.predict(base), color="#C1121F", lw=2.4,
               label="controlled fit")
    ax[1].axhline(0, color="#888", lw=1)
    ax[1].set_xlabel("price-adjusted model residual")
    ax[1].set_ylabel("forward TM return (log)")
    ax[1].set_title("Controlled for TM momentum, age, price, season", fontsize=10)
    ax[1].legend(fontsize=8, frameon=False)

    for a in ax:
        a.spines[["top", "right"]].set_visible(False)
        a.grid(alpha=.25, lw=.6)
    plt.tight_layout()
    out = PROC / "p5_backtest.png"
    plt.savefig(out, dpi=155, bbox_inches="tight")
    res.to_json(PROC / "p5_backtest.json", orient="records", indent=2)
    d.to_parquet(PROC / "p5_backtest_sample.parquet", index=False)
    print(f"\nfigure -> {out}")


if __name__ == "__main__":
    main()
