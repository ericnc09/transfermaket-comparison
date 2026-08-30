# Model Card — Transfer Value Engine (Big 5 outfield, cold-start / update)

Gradient-boosted valuation models for outfield footballers in Europe's Big 5
leagues, trained to reproduce Transfermarkt market values and to surface where
they disagree with them.

Two variants ship, and **they are meant to be read together**:

| Variant | Prior Transfermarkt value | Purpose |
|---|---|---|
| `coldstart` | **excluded** | The disagreement engine. Infers worth from football alone, so its residuals carry information. |
| `update` | included | The accuracy view. Answers "what will Transfermarkt say next?" |

Quoting `update` alone overstates what the model knows; quoting `coldstart` alone
understates how well the next valuation can be predicted.

---

## Model overview

- **Algorithm**: LightGBM (`predict()` path, chosen for clean SHAP support) and
  CatBoost (leaderboard best). The two are statistically indistinguishable here
  — 0.401 vs 0.408 cold-start log MAE against a fold-to-fold sd of ~0.09.
- **Target**: `log1p(value_eur)`. Predictions are exponentiated back to euros.
- **Alignment**: features from season *t* predict the Transfermarkt valuation
  published at the start of season *t+1*. Nothing dated on or after the label
  enters the feature matrix.
- **Features**: 175 (`coldstart`) / 178 (`update`).
- **Intervals**: split conformal at 80% nominal, calibrated on a
  player-disjoint slice, residuals taken in log space so the euro band is
  multiplicative.
- **Artifacts**: `data/processed/models/conformal_lgbm_{coldstart,update}.joblib`.
- **License**: MIT (code). Underlying data is third-party — see [Data](#training-data).

---

## Intended use

- **Finding pricing disagreements.** Rank players by price-adjusted residual to
  see who the model values differently from Transfermarkt, and read the SHAP
  attribution for why.
- **Research baseline.** A reproducible cold-start valuation model with a
  documented temporal protocol, for work benchmarking against Transfermarkt.
- **Exploratory scouting support.** A prompt for a human to look closer at a
  player, alongside the comparables list.

## Out-of-scope use

- **Any transfer, contract, or financial decision.** The model reproduces a
  crowd-sourced estimate, not intrinsic worth, and has no access to fees,
  wages, agent terms, or negotiations.
- **Betting or trading signals.** The forward backtest measures convergence
  toward the model's own opinion, not profit, and carries no transaction model.
- **Goalkeepers.** Excluded from training entirely — they are valued on
  different features and pooling them degrades both.
- **Players outside the Big 5**, outside 2017-18 → 2021-22, or below 600
  minutes in a season. All are unrepresented in training.
- **Judging individuals.** A low valuation reflects what the features capture,
  which is not the same as a judgement about a footballer.

---

## Training data

| Source | Role | Access |
|---|---|---|
| FBref (Opta-derived) | 9 season stat tables | [worldfootballR_data](https://github.com/JaseZiv/worldfootballR_data) mirror — fbref.com is behind a Cloudflare challenge |
| Transfermarkt | Labels, bio, contract | `ceapi` value-history endpoint, crawled at 1.2 s/request |

**Panel: 8,247 player-seasons over 3,060 unique players, five seasons
(2017-18 → 2021-22).** Outfield only, ≥600 minutes.

| League | n | | Position | n |
|---|---|---|---|---|
| Serie A | 1,735 | | CB | 1,791 |
| La Liga | 1,718 | | FB | 1,499 |
| Premier League | 1,684 | | CM | 1,247 |
| Ligue 1 | 1,640 | | W | 1,194 |
| Bundesliga | 1,470 | | ST | 1,146 |
| | | | DM | 795 |
| | | | AM | 575 |

Label range €0.1m – €200m, median €7.0m.

**Exclusions, all data-driven rather than hardcoded:** stale label snapshots
(`stale_label_seasons()`), part-played seasons (`partial_seasons()`),
goalkeepers, €0 valuations, and rows below the minutes threshold.

---

## Evaluation

Rolling-origin cross-validation: each season is predicted by a model trained
only on earlier seasons. Random splits are never used — they would leak future
valuations into the past.

### Leaderboard (mean over 4 folds)

| Model | log MAE | log R² | MedAE €m | ±30% | NDCG@100 |
|---|---|---|---|---|---|
| CatBoost `[update]` | 0.285 | 0.906 | 1.42 | 61.8% | 0.935 |
| **CatBoost `[coldstart]`** | **0.401** | 0.813 | 2.01 | 48.4% | 0.883 |
| *carry forward prior TM value* | *0.497* | *0.641* | *2.33* | *42.4%* | *0.910* |
| EBM `[coldstart]` | 0.470 | 0.682 | 2.13 | 44.6% | 0.861 |
| global median | 1.010 | −0.017 | 4.60 | 15.9% | 0.146 |

**The cold-start model beats carry-forward** (0.401 vs 0.497) — it predicts the
next valuation better than Transfermarkt's own previous valuation does, without
ever seeing a Transfermarkt value.

Note that carry-forward still holds the second-best NDCG@100 (0.910): **nothing
ranks the most expensive players better than leaving last year's number alone.**

### Interval calibration (nominal 80%)

| Method | Realised coverage |
|---|---|
| conformal `[coldstart]` | **0.792** |
| conformal `[update]` | **0.797** |
| LightGBM quantile `[coldstart]` | 0.622 |
| LightGBM quantile `[update]` | 0.668 |

Use the conformal intervals. The quantile objective under-covers badly enough
that its "p10–p90" band would be misleading.

### Performance by segment (cold-start, out-of-sample mean |log error|)

| Position | | League | | TM value quintile | |
|---|---|---|---|---|---|
| AM | 0.360 | Premier League | 0.357 | €1.5m | 0.422 |
| FB | 0.370 | Serie A | 0.365 | €4m | 0.353 |
| CM / ST / W | 0.376 | Ligue 1 | 0.366 | €8m | 0.356 |
| CB | 0.385 | Bundesliga | 0.389 | €15m | 0.371 |
| **DM** | **0.408** | **La Liga** | **0.417** | **€35m** | **0.385** |

Worst cases: defensive midfielders, La Liga, and the cheapest quintile (where
Transfermarkt values are coarse and quantised).

---

## Forward backtest

Does Transfermarkt move toward the model after a disagreement? Over 4,591
player-seasons and 2,359 players:

**β = +0.187 (s.e. 0.018, p = 2.8e-25)**, controlling for Transfermarkt's own
recent move, age, price level and season, with errors clustered by player. The
market closes about a fifth of the gap within a year.

Positive and significant in all four seasons independently, and in the subsample
where Transfermarkt has *not* recently moved the value (β = +0.213, p = 0.0004)
— which is what separates the result from mean reversion.

**This does not show the model is right and Transfermarkt wrong about intrinsic
worth.** Transfermarkt value is the only ground truth in this design, so the test
measures convergence toward the model, not toward realised fees.

---

## Known blind spots

**The model sees football. It does not see anything else.** Everything below
follows from that.

- **Off-field events.** The clearest case in the data: Mason Greenwood, 2021-22.
  Transfermarkt priced him at €5m after a suspension; the model, seeing only his
  football, said €41m and flagged him the single most undervalued player in the
  panel. The model is not wrong about the football and not right about the
  player. Treat large residuals as *questions*, never conclusions.
- **Reputation and brand.** Established stars are systematically marked down —
  Ronaldo, Icardi, Coutinho all appear among "overvalued". Transfermarkt prices
  a name; the model prices a season.
- **Injury and availability.** No injury history is ingested. A player returning
  from a long absence has low minutes, which the model reads as low value.
- **Potential.** Young players are priced by the market on projection. The model
  has only production, so it disagrees most where potential dominates — Haaland
  2021-22 at €79m model vs €150m Transfermarkt.
- **Transfer context.** Contract length is included; release clauses, wage
  demands, agent relationships and selling-club urgency are not.

## Known biases and statistical caveats

- **Residuals must be read within a price stratum.** Every regression shrinks
  toward the mean: raw residuals run from **+0.360** in the cheapest decile to
  **−0.338** in the dearest. Ranking on raw residuals returns cheap players as
  "undervalued" by construction. The shipped leaderboard is price-stratified.
- **Survivorship.** The panel only contains players who cleared 600 minutes in a
  Big 5 league. Players who left, were injured, or never broke through are
  absent, so the model is not calibrated for them.
- **The label is a crowd estimate.** Transfermarkt values are community-sourced
  and moderator-adjusted. The model's ceiling is the quality of that consensus.
- **Effective sample is players, not rows.** 8,247 rows over 3,060 players;
  the same footballer recurs across seasons. All CV is player-grouped and the
  backtest clusters standard errors by player.
- **COVID.** The 2019-20 and 2020-21 seasons carry a market-wide markdown.
  Season effects absorb it in the backtest; single-season predictions from that
  era should be read with that in mind.
- **2022-23 is unusable** — the source stopped updating mid-season (~13 matches).
  The model cannot score that season without imputation.

---

## Reproducing

```bash
python -m venv .venv && ./.venv/bin/pip install -r requirements.txt
./.venv/bin/python scripts/00_setup_openmp.py        # macOS: LightGBM/XGBoost without brew
./.venv/bin/python scripts/01_ingest.py              # ~30 s
./.venv/bin/python scripts/05_fetch_tm_history.py    # ~65 min, rate-limited, resumable
./.venv/bin/python scripts/03_build_panel.py
./.venv/bin/python scripts/07_generate_predictions.py
./.venv/bin/python scripts/09_backtest.py
./.venv/bin/python -m pytest tests/ -q               # 34 gates
```

Data files are build artefacts and are not committed.

## Using it

```python
from src.predict import search, lookup
v = lookup("Mason Greenwood", 2022)
print(v)            # model value, interval, TM value, delta
v.why               # SHAP attribution, as multiplicative effects
v.comparables       # five most similar players
```

---

## Changelog

| Version | Change |
|---|---|
| P5 | Forward backtest: β = +0.187 controlled. €0 forward values excluded. |
| P4 | Out-of-sample predictions, SHAP, similarity, price-stratified residuals. Deflation dropped (measurably harmful). `Season_End_Year`/`Born` removed as features. |
| P3 | Labels repaired from Transfermarkt's dated history; panel 5,684 → 8,247 rows. Optuna pass — no material gain. |
| P2 | Baselines + gradient boosting. Aggregate leakage and stale labels found. |
| P1 | Feature matrix, panel, leakage gates. |
| P0 | Entity resolution: 98.8% across the Big 5. |

Full decision record with rationale: [progress.md](progress.md).
Design and build log: <https://claude.ai/code/artifact/7f2391b1-149f-4f3b-bbf0-37063d6d38dd>
