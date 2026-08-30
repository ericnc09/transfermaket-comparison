# Progress log

Living record of this project: the plan, every decision and why, and what has
actually been built. **Updated at the start of every working session and before
every commit or push.**

Last updated: 2026-08-30 · after P5 (forward backtest)

---

## Status

| Phase | Scope | State |
|---|---|---|
| **P0** | Kill-risk spike — prove the FBref ↔ Transfermarkt join | ✅ Complete |
| **P1** | Data platform — feature matrix, panel, leakage gates | ✅ Complete |
| **P2** | First models — baselines + GBM, temporal eval | ✅ Complete |
| **P3** | Model zoo — full tier list, quantiles, ensemble, tuning | ✅ Complete |
| **P4** | Explain & compare — SHAP, similarity, residual leaderboard | ✅ Complete |
| **P5** | Research — forward backtest | ✅ Backtest complete · model card outstanding |

Full system design: <https://claude.ai/code/artifact/7f2391b1-149f-4f3b-bbf0-37063d6d38dd>

---

## What this project is

A machine-learning valuation model for Big 5 outfield footballers, trained to
reproduce Transfermarkt market values and to surface where it disagrees with them.

**Target:** `log1p(Transfermarkt market value)`
**Alignment:** stats from season *t* predict the valuation published at the start of
season *t+1*. Nothing dated on or after the label enters the feature matrix.
**Output:** model value, prediction interval, TM value, delta, SHAP attribution.

---

## Decisions on record

### Round 1 — framing

| Question | Answer | Consequence |
|---|---|---|
| What does the model predict? | Show the difference between our model and Transfermarkt | Target is market **value**, not fee — see correction below. Framing A (replicate), residual as headline output. |
| Use prior TM value as a feature? | **Include it** | Model hugs TM closely (`corr(log prior, log label) = 0.925`). A `coldstart` variant was added alongside so deltas still mean something. |
| How to source data? | **Everything scraped** | Revised: FBref is Cloudflare-blocked, so an open mirror is used instead. See P0 findings. |
| What is v1? | **Notebooks + leaderboard** | No Streamlit app in v1; `predict()` callable from a notebook satisfies "plug any player in". |

**Correction made during planning.** Every Transfermarkt player has a *market value*,
a community-sourced estimate. Only players who actually transferred have a *fee*, and
most of those are undisclosed, loans, or frees. Market value is therefore the label;
fees are demoted to an optional P5 validation set.

### Round 2 — scope

| Question | Answer | Consequence |
|---|---|---|
| Panel width | **Big 5, 2017-18 → 2024-25** | Revised to **2017-18 → 2022-23** — the mirror's coverage ceiling. |
| Snapshot cadence | **Every TM update (~3×/year)** | Revised to **one per season** — the mirror carries season-start values only. |
| Inclusion | **Outfield only, ≥600 min** | No goalkeepers, no low-minute players, no SoFIFA/EA ratings. |
| Repo and stack | **`transfermarkets/`, polars + duckdb** | Fresh git init; pandas at the model boundary. |

### Round 3 — after P0 findings

| Question | Answer |
|---|---|
| Six clean seasons, or eight with two feature eras? | **Six clean seasons, one consistent feature matrix.** When the point is comparing model behaviour against Transfermarkt, a stable feature set beats recency. |

### Round 4 — after the stale-label repair

| Question | Answer | Consequence |
|---|---|---|
| Panel unit, now that real dated valuations restore the ~3×/year cadence? | **One row per player-season** | Trailing-window features and every existing test stay valid. The scraped history is used to pick one precise post-season label rather than to multiply rows. Avoids ~25,000 heavily autocorrelated rows over 2,415 players. |

### Two model variants

Both run off one config flag over the same pipeline.

| Variant | Prior TM value | Job |
|---|---|---|
| `update` | Included | Accuracy leaderboard. Answers "what will TM say next?" |
| `coldstart` | Excluded | Disagreement engine. Answers "what is this player worth on the football alone?" |

---

## P0 — kill-risk spike ✅

**Goal:** prove FBref player-seasons can be joined to Transfermarkt valuations before
investing in anything else.

### Result

| Population | Match rate |
|---|---|
| Big 5, outfield, ≥600 min, 2017-18 → 2022-23 | **98.8%** (9,860 / 9,976) |
| Premier League | 99.1% |
| Worst league — La Liga | 97.7% |
| Worst league-season — La Liga 2022-23 | 94.1% (truncated source) |

94.3% of links resolve at Tier A (exact name + birth year). Tier C (birth year + fuzzy
name) carries 4.3% and catches the word-order and nickname variants: *Son Heung-min* vs
*Heung-min Son*, *Pierre Højbjerg* vs *Pierre-Emile Højbjerg*, *Martinelli* vs
*Gabriel Martinelli*.

Correctness spot-checked, not just coverage: Salah €90m, Kane €90m, De Bruyne €85m,
Rodri €80m — all genuine 2022 Transfermarkt values.

Signal confirmed on the PL panel (n=1,550): age **ρ=−0.467**, npxG+xAG/90 **ρ=+0.319**,
minutes **ρ=+0.301**. Median value rose €12.0m → €20.0m across five seasons.

### Three findings that changed the plan

1. **FBref is behind a Cloudflare JS challenge.** Direct scraping returns 403.
   Defeating bot protection is out of scope. Stats come from the
   [worldfootballR_data](https://github.com/JaseZiv/worldfootballR_data) mirror, which
   publishes the same Opta-derived tables openly — *and* Transfermarkt valuations with
   full date of birth, which is what made the join so clean. Better than the original
   plan: 10 MB in 30 seconds instead of a 6–9 hour polite crawl.
2. **The mirror stopped updating in late 2022.** Panel is six seasons, not eight.
3. **Transfermarkt values are one snapshot per season**, not ~3/year.

---

## P1 — data platform ✅

**Built:** `src/features/{definitions,build,manifest}.py`, `scripts/03_build_panel.py`,
`tests/test_panel.py`.

### The panel

| | |
|---|---|
| Rows (eligible + labelled) | **5,684** |
| Unique players | **2,415** |
| Features | 177 cold-start / 180 update |
| Seasons | 2017-18 → 2020-21 (four label-able) |
| Leagues | Big 5, 1,050–1,185 rows each |
| Store | `data/processed/market.duckdb` + parquet |

> Originally 7,077 rows over five seasons. P2 found the 2021-22 label snapshot to
> be stale and excluded it — see [P2](#p2--first-models-) below.

### What the builder does

- Merges nine FBref stat tables on `(season, squad, competition, player_url)`.
- **Aggregates mid-season transfers rather than truncating them.** 638 player-seasons
  spanned two clubs; counting stats are summed so a January mover keeps his whole
  season, rate columns take a minutes-weighted mean, and context comes from the club
  he played most minutes for. The original spec said keep the highest-minutes row —
  that would have discarded half a season for 9% of the panel.
- Converts 61 counting stats to per-90 rates; adds ratios (shot accuracy, aerial win %,
  dribble success), overperformance (`G−xG`, `A−xAG`), and `npxG+xAG/90`.
- Attaches Transfermarkt label, prior valuation, date of birth, height, foot, contract
  expiry, date joined. **Age and contract-months-left are computed against the label
  date**, not the season, because that is what TM was actually pricing.
- Context: squad total market value and its rank in the league, league median value,
  and target deflation by that median so the model learns quality, not inflation.
- True *t−1* and *t−2* lags, joined explicitly on `(player, season−k)` so a missing
  season yields null rather than a silently shifted row.

151 of 180 features are ≥99% populated. The 11 below 50% are all `lag2` columns,
structurally absent for a player's first two observed seasons.

### Leakage gates — `tests/test_panel.py`

- No feature correlates >0.98 with the target.
- `prior_value` verified **against the raw source, row by row on a random sample**, to
  come from season *t−1*, and the label from season *t+1*.
- Lag columns verified to equal the same player's value one season earlier.
- The cold-start variant provably excludes every prior-valuation column.
- No target column appears in either feature set.

One test failed initially and was a genuine design finding: lags are drawn from the
*full* panel, so a player whose previous season fell below the 600-minute bar still has
a valid lag (Jonathan Silva, 112 minutes in 2017-18). That breakout signal is exactly
what the feature is for, so the test's reference frame was fixed, not the behaviour.

### Data-quality items found

- **Mid-season transfers are double-counted in Transfermarkt** — a player appears under
  both squads in the season he moves. The builder deduplicates on `(player, season)`.
- **The 2022-23 Transfermarkt scrape is truncated** — 549 PL rows against ~780 in other
  seasons. Sole cause of the 94–96% match rates that season. Affects features only,
  never labels, since 2022-23 carries no *t+1* value.
- **`pressures` and `prog_carries` vanish in 2022-23** — FBref restructured them. 100%
  populated in every labelled season, so training is unaffected; scoring a 2022-23
  player would need imputation.

---

## P2 — first models ✅

**Built:** `src/eval/{splits,metrics}.py`, `src/models/{baselines,gbm}.py`,
`scripts/04_train.py`.

Split: train 2017-18 → 2018-19 (2,836) · validate 2019-20 (1,432) · test 2020-21 (1,416).

### Test leaderboard — 2020-21, held out

| Model | log MAE | log R² | MAE €m | MedAE €m | ±30% | Spearman | NDCG@100 |
|---|---|---|---|---|---|---|---|
| **CatBoost [update]** | **0.323** | **0.882** | 3.62 | 1.60 | 57.6% | 0.940 | 0.939 |
| HistGBM [update] | 0.323 | 0.881 | 3.65 | 1.65 | 58.2% | 0.938 | 0.934 |
| *carry forward prior TM value* | *0.384* | *0.832* | *4.21* | *2.00* | *45.1%* | *0.920* | *0.925* |
| **CatBoost [coldstart]** | 0.488 | 0.741 | 5.46 | 2.78 | 38.0% | 0.869 | 0.864 |
| HistGBM [coldstart] | 0.488 | 0.748 | 5.59 | 2.53 | 38.5% | 0.863 | 0.839 |
| log-linear (age, minutes, G+A) | 0.735 | 0.470 | 8.34 | 4.12 | 23.5% | 0.666 | 0.690 |
| median by position × age | 0.888 | 0.227 | 10.08 | 5.20 | 18.9% | 0.435 | 0.286 |
| global median | 1.026 | −0.000 | 10.72 | 5.40 | 17.5% | — | 0.161 |

**How to read this.** The `update` models beat carry-forward by 16% on log MAE
(0.323 vs 0.384) and lift ±30% hit rate from 45% to 58%. That *gap* is the model's
contribution, not the R² of 0.88. The `coldstart` model reaching R² 0.741 having
never seen a Transfermarkt value is the more meaningful result — it is inferring
worth from football alone.

CatBoost and HistGBM are statistically indistinguishable here. LightGBM and XGBoost
need the OpenMP runtime (`brew install libomp`), which is not installed on this
machine; both can join the zoo in P3 if that changes.

### Segment findings

- **Under-24s are hardest** — log MAE 0.378 against ~0.30 for every older band.
  Young players are priced on potential, which production stats do not carry.
- **The expensive tail is expensive to miss** — MedAE €7.67m in the top value
  quartile against €0.53m in the bottom, exactly as anticipated.
- **Premier League is the best-modelled league** (log MAE 0.258, 72.8% within ±30%);
  Ligue 1 the worst (0.334, 52.3%).
- Position differences are mild; DM is weakest on hit rate (50.0%).

### Two defects found and fixed during P2

**1. Aggregate leakage in the context features.** `squad_value_eur` and
`league_median_value` were computed from the *same* Transfermarkt snapshot the label
comes from, so each contained the player's own label. James Tarkowski's €25m label sat
inside his €132m squad total. The >0.98 correlation test could not catch it — diluted
across a squad the leak shows up as r≈0.03. Both now come from the **prior** season's
snapshot, and `test_context_aggregates_predate_the_label` verifies it against the raw
source.

**2. The 2021-22 Transfermarkt label snapshot is stale.** It is a **99.5% copy** of the
previous season; every other season pair moves 5–22% of players. Left in, the
carry-forward baseline scored a perfect R²=1.000 — labels *were* the priors. Detection
is now data-driven (`stale_label_seasons()`, threshold 90% unchanged) rather than a
hardcoded year, so a future stale snapshot is caught automatically.

**Cost:** the panel dropped from 7,077 rows over five seasons to 5,684 over four, and
the training window is now two seasons, which makes every `lag2` feature entirely null
in training. A variance guard drops degenerate columns at fit time rather than
hardcoding the exclusion.

---

## The data repair ✅

The stale-label defect was fixed at the source rather than worked around. Transfermarkt's
own `ceapi/marketValueDevelopment` endpoint returns each player's **full dated value
history**; `scripts/05_fetch_tm_history.py` crawled all **3,175** panel players at 1.2 s
per request (~65 min, 100% success, 23 MB cached).

| | Before | After |
|---|---|---|
| Rows | 5,684 | **8,247** |
| Unique players | 2,415 | **3,060** |
| Label-able seasons | 4 | **5** (2017-18 → 2021-22) |
| Label source | mirror snapshots | **100% scraped, dated** |
| `label == prior` | 99.5% in the bad season | 6–20% in every season |

Labels are now defined precisely: for Season_End_Year *Y*, the label is the first
valuation dated on or after 1 June *Y*, and the prior is the last dated before
1 August *Y−1*. Both are verified against the raw history by
`test_scraped_label_is_dated_after_the_season`.

### Three further defects the gates caught

**2022-23 is a partial season.** The mirror stopped updating in November 2022, so its
2022-23 rows hold ~13 matches and 1,170 minutes against 3,420 for a full season. Per-90
rates from a third of a season are far noisier and every volume feature is on a
different scale, while the label is still a full post-season valuation. Detected
data-driven by `partial_seasons()` and excluded.

**1,321 rows had no position.** Scraped labels admit rows the mirror never covered in
that season, leaving bio fields null. Position, height and foot do not change, so they
are backfilled from the player's modal values across every season the mirror does have.
This also cut missing `squad_value_prior` from 2,065 rows to 879.

**16 rows carried a €0 valuation** — an absence of data, not a price. Excluded; this
also fixed a stacker round-trip test that was tripping on the clipping boundary.

---

## P3 — model zoo ✅

Rolling-origin CV, 4 folds, on the repaired panel. Figures below are **post-tuning**;
the tuning changed almost nothing, which is itself the finding — see below.

### Built

| Component | File | Purpose |
|---|---|---|
| Ridge / ElasticNet + age splines | `src/models/linear.py` | Interpretable floor. Age enters via a natural spline basis, not a raw term. |
| Explainable Boosting Machine | `src/models/ebm.py` | Near-GBM accuracy with a readable shape function per feature — the age curve comes straight off the model. |
| Quantile trio (p10/p50/p90) | `src/models/quantile.py` | LightGBM quantile objective, with monotonicity enforced against quantile crossing. |
| Split-conformal wrapper | `src/models/quantile.py` | Distribution-free coverage around any point model. Calibrated on a **player-disjoint** slice; residuals taken in log space so the euro band is multiplicative. |
| Stacked ensemble | `src/models/ensemble.py` | Ridge meta-learner over player-grouped out-of-fold base predictions. |
| Optuna harness | `src/models/tuning.py` | Search spaces for all five families; objective is mean log MAE over player-grouped CV. |
| Orchestrator | `scripts/06_tune_and_evaluate.py` | Tune → rolling-origin evaluate → interval calibration. |
| Contract tests | `tests/test_models.py` | Units, interval ordering, conformal coverage, no-future-leakage in the fold generator. |

### Leaderboard — mean over 4 rolling-origin folds

| Model | log MAE | ±sd | log R² | MedAE €m | ±30% | ρ | NDCG@100 |
|---|---|---|---|---|---|---|---|
| **Stacked [update]** | **0.285** | 0.046 | 0.906 | 1.42 | 61.7% | 0.959 | 0.947 |
| LightGBM [update] | 0.288 | 0.040 | 0.905 | 1.43 | 61.2% | 0.957 | 0.943 |
| CatBoost [update] | 0.289 | 0.049 | 0.904 | 1.46 | 60.8% | 0.957 | 0.938 |
| HistGBM [update] | 0.291 | 0.043 | 0.903 | 1.45 | 61.1% | 0.955 | 0.938 |
| XGBoost [update] | 0.298 | 0.050 | 0.900 | 1.51 | 59.2% | 0.956 | 0.942 |
| EBM [update] | 0.340 | 0.055 | 0.812 | 1.51 | 58.3% | 0.953 | 0.924 |
| **CatBoost [coldstart]** | **0.403** | 0.089 | 0.807 | 2.01 | 48.2% | 0.910 | 0.885 |
| Stacked [coldstart] | 0.403 | 0.089 | 0.800 | 1.96 | 48.4% | 0.912 | 0.891 |
| LightGBM [coldstart] | 0.413 | 0.083 | 0.809 | 2.07 | 46.3% | 0.901 | 0.880 |
| ElasticNet+splines [update] | 0.453 | 0.071 | 0.643 | 2.11 | 47.1% | 0.933 | 0.930 |
| EBM [coldstart] | 0.474 | 0.076 | 0.670 | 2.14 | 44.9% | 0.898 | 0.864 |
| *carry forward prior TM value* | *0.497* | *0.051* | *0.641* | *2.33* | *42.4%* | *0.852* | *0.910* |
| Ridge+splines [coldstart] | 0.565 | 0.067 | 0.487 | 2.63 | 38.0% | 0.879 | 0.868 |
| log-linear (age, minutes, G+A) | 0.725 | 0.014 | 0.463 | 3.67 | 25.3% | 0.657 | 0.712 |
| global median | 1.010 | 0.015 | −0.017 | 4.60 | 15.9% | — | 0.146 |

**The headline changed once the labels were real.** Carry-forward fell from 0.402 to
**0.497**, because a large share of the old labels were stale copies of the prior value
and were flattering it. Two consequences:

1. The `update` models now beat carry-forward by **43%** (0.285 vs 0.497), not the 10%
   the broken panel suggested.
2. **The cold-start model beats carry-forward outright** (0.403 vs 0.497). A model that
   has never seen a Transfermarkt value predicts the next valuation better than
   Transfermarkt's own previous valuation does. That is the result the whole project
   was built to test, and it is the one to write up.

Carry-forward still holds a strong NDCG@100 (0.910): nothing ranks the most expensive
players quite like leaving last year's number alone.

**Ridge and ElasticNet look better than they are.** Competitive log MAE (0.453) but
MAE €10.1m — being linear in log space, they blow up multiplicatively on the expensive
tail. Useful as an interpretable floor, not as a predictor.

### The Optuna pass bought nothing — and that is the result

5.7 hours of compute across 10 tuning jobs (LightGBM ~19/29 min per variant, XGBoost
~24/29, CatBoost ~51/47 even on a reduced budget, HistGBM ~24/19, EBM ~11/10).

| | log MAE gain |
|---|---|
| Mean gain across 20 models | **+0.0017** |
| Best single gain (XGBoost cold-start) | **+0.0094** |
| Mean fold-to-fold standard deviation | **0.060** |

The largest improvement is **11% of one standard deviation**. Every other gain is
smaller still, and three models got marginally worse. Tuned hyperparameters are kept
(`data/processed/p3_best_params.json`) because they cost nothing to carry, but no
conclusion in this project rests on them.

**This confirms the prediction made before the pass ran: the data is the binding
constraint, not the hyperparameters.** With 8,247 rows over 3,060 players and a label
that is itself a crowd-sourced estimate, the ceiling is set by the panel. The practical
implication for P4/P5 is to stop buying accuracy and start buying *explanation* —
more compute on this model family is wasted effort.

### Interval calibration — nominal 80%

| Method | PICP | median width €m | width ratio |
|---|---|---|---|
| LGBM quantile [coldstart] | **0.622** | 6.49 | 2.31 |
| conformal 80% [coldstart] | **0.826** | 9.99 | 3.31 |
| LGBM quantile [update] | **0.668** | 4.28 | 1.71 |
| conformal 80% [update] | **0.839** | 6.46 | 2.26 |

Confirmed at full scale: LightGBM's quantile objective under-covers badly — a
"p10–p90" band that actually contains the truth 62% of the time would be a lie. Split
conformal lands within four points of nominal in both variants, at the cost of a wider
band. **Conformal is the interval method.**

### A bug found and fixed

The stacker initially scored **0.614** — worse than every one of its components, which
is close to impossible for a ridge meta-learner and so was a bug rather than a result.
Base predictions were being fed in as `log1p(euros)` while the meta-learner's target
was `log1p(value / league_median_prior)`, so the ridge had to absorb a per-row deflator
it could not see. With both sides in the same space it scores **0.463**, between its
components as expected. `test_stacker_target_space_roundtrip` now pins the conversion.

---

## P4 — explain and compare ✅

**Built:** `src/explain/{shap_utils,similarity}.py`, `src/predict.py`,
`scripts/07_generate_predictions.py`, `scripts/08_residuals.py`.

6,546 out-of-sample predictions — every season scored by a model trained only on
earlier seasons — with conformal intervals.

| Variant | log MAE | MedAE €m | ±30% | interval coverage |
|---|---|---|---|---|
| coldstart | 0.379 | 1.90 | 50.2% | **0.792** |
| update | 0.277 | 1.40 | 64.6% | **0.797** |

Coverage lands on the nominal 0.80 in both variants.

### The lookup — the original goal, delivered

```python
from src.predict import search, lookup
v = lookup("Mason Greenwood", 2022)
```
```
Mason Greenwood  ·  2021-22  ·  Premier League  ·  W  ·  age 20.7  ·  1,271 min
  model (coldstart) : EUR 41.1m   [EUR 23.6m – EUR 71.6m]
  transfermarkt     : EUR 5.0m
  delta             : EUR 36.1m  (723% undervalued by the model)
```

That case is the clearest illustration of the cold-start model's blind spot:
Transfermarkt crashed his value for off-field reasons the model cannot see, so it
values the football alone. Useful to understand, not a trading signal.

### Three defects found and fixed in P4

**Deflation was actively hurting — the design assumption was wrong.** Measured head to
head: cold-start log MAE **0.374 undeflated against 0.427 deflated**, roughly a standard
deviation, with lower fold-to-fold variance too. The divisor was a noisy per-row
quantity the model then had to undo, while league level and inflation were already
available as ordinary features. The target is now `log1p(value_eur)` directly, which
also removes an entire class of bug.

That class had already bitten: deflating by the raw median of every Transfermarkt entry
put Serie A 2018-19 at €0.8m against €1.5–2.4m for the rest, because that snapshot
listed far more fringe players. The deflated target for that one cell sat at 7.5 where
every other cell was near 2–3, so the model marked the **entire league** down —
Ronaldo at €11m. It was caught because the "most overvalued" list came back 15-for-15
Serie A 2019, which is not a finding, it is a bug.

**`Season_End_Year` and `Born` were features.** Under a temporal split every test season
is later than anything in training, so a tree can only extrapolate whatever the last
training season taught it; birth year additionally lets the model recover the season
index. Both removed. Metric effect was inside noise, but the model is now correct.

**The residual leaderboard was ranking price, not players.** Every regression shrinks
toward the mean, so raw residuals ran from **+0.360** in the cheapest decile to
**−0.338** in the dearest, against an overall mean of −0.034. Ranking on them returned
cheap players as "undervalued" and expensive ones as "overvalued" by construction.
Residuals are now compared **within Transfermarkt price strata**. Once corrected, the
systematic disagreements are small: La Liga −0.083, Bundesliga +0.054, DM −0.063,
ST +0.044.

**And one leak in my own analysis.** A scratch comparison of deflated vs raw targets
reported R²=0.998 — because a temporary `log_value_raw` column landed on the frame and
`feature_columns` picked it up as a feature, so the model was predicting the target
from itself. Caught by the implausibility of the number. The real comparison is above.
The lesson is that the manifest's `TARGETS` set is what prevents this, and ad-hoc
columns bypass it.

---

## P5 — the forward backtest ✅

**Built:** `forward_values()` in `src/features/labels.py`, `scripts/09_backtest.py`.

The question the whole project was built to answer: when the cold-start model disagrees
with Transfermarkt, does Transfermarkt subsequently move *toward* the model?

Sample: **4,591 player-seasons, 2,359 players, four seasons.** For each row, the
Transfermarkt valuation one year after the label date (±120 days, else no row).

### Why the naive version of this test is worthless

Let *V* be the Transfermarkt value and suppose it carries transient noise *e* —
Transfermarkt has overshot. The residual (log *M* − log *V*) then contains −*e*, and the
forward return (log *V′* − log *V*) also contains −*e*, because the overshoot reverts.
**Any model that merely fails to replicate Transfermarkt's noise will appear to predict
its future movements.** That is mean reversion, not skill, and a raw correlation cannot
tell the two apart.

### Result

| Specification | β on residual | s.e. | p |
|---|---|---|---|
| raw, no controls | +0.231 | 0.020 | ~0 |
| + TM momentum | +0.232 | 0.019 | ~0 |
| **+ age, price, season** | **+0.187** | **0.018** | **2.8e-25** |

Standard errors clustered by player. A 1.0 log-point disagreement maps to a **+0.187
log-point** move in the following year's Transfermarkt value — the market closes about
**19% of the gap within a year**. Raw top-minus-bottom decile spread is +0.332 log
points (+39% in value terms), but that figure includes the mean reversion and is not the
claim.

### It survives the checks that matter

- **Every season independently**: β = +0.275, +0.173, +0.206, +0.137 (all p < 0.001).
  Not a single-season fluke that a fixed effect is hiding.
- **Where Transfermarkt has *not* recently moved the value** (|momentum| < 0.05, n=532):
  β = **+0.213, p = 0.0004**. Pure mean reversion needs a recent move to revert; the
  signal is undiminished where there is none. This is the strongest single piece of
  evidence against the reversion explanation.

### A defect the figure caught

The first run gave β = +0.243, but the scatter showed a cluster at −12 to −16 log
points: **29 rows (0.63%) with a €0 forward value** — players Transfermarkt stopped
pricing, where `log1p(0)` turns a €10m player into a −16 log-point "return". Excluding
them *lowered* the coefficient to +0.187 while raising the t-statistic from 7.2 to 10.4
and R² from 0.117 to 0.234. It also flipped the key robustness test: the low-momentum
subsample went from p = 0.15 to p = 0.0004. The outliers had been inflating the estimate
and destroying the precision of the check that mattered most.

### What this does and does not show

It shows the cold-start model carries information about future Transfermarkt movements
that is not explained by mean reversion, age, price level, or season. It does **not**
show the model is right and Transfermarkt wrong about intrinsic worth — Transfermarkt
value is the only ground truth here, and the test measures convergence toward the model,
not toward realised fees. The single control for reversion is one lag of Transfermarkt's
own move; a richer autoregressive control would strengthen the claim further.

---

## Deviations from the original spec

| Original plan | Actual | Why |
|---|---|---|
| Scrape FBref, Understat, TM, SoFIFA | Open mirror for FBref + TM | Cloudflare JS challenge; bot protection not bypassed |
| 6–9 hour polite crawl | 30-second download, 10 MB | Mirror ships pre-extracted parquet-able tables |
| 8 seasons, 2017-18 → 2024-25 | 6 seasons, 2017-18 → 2022-23 | Mirror's coverage ceiling |
| ~3 snapshots/year, ~45,000 rows | 1/season, 7,077 labelled rows | Mirror carries season-start values only |
| ~6,500 effective players | 2,688 unique players | Follows from the above; MLP demoted to optional |
| football-data.org as fixture spine | Dropped | Free tier has no minutes, xG, or defensive actions |
| Mid-season transfer: keep max-minutes row | Aggregate across clubs | Truncating loses half a season for 9% of rows |
| 5 label-able seasons, 7,077 rows | 4 seasons, 5,684 rows | 2021-22 TM snapshot is a 99.5% copy of 2020-21 |
| LightGBM as primary GBM | CatBoost + sklearn HistGBM | LightGBM/XGBoost need OpenMP; not installed, and not worth a system change |
| Squad/league context from current snapshot | Prior-season snapshot | Contemporaneous aggregates contain the player's own label |
| Deflate the target by league median | **No deflation** — raw `log1p(value_eur)` | Measured worse (0.427 vs 0.374 cold-start) and caused a whole-league failure |
| Season index available to the model | `Season_End_Year`, `Born` excluded | Under a temporal split the model can only extrapolate a time trend it cannot know |
| Rank residuals directly | Rank within price strata | Raw residuals encode regression to the mean, not disagreement |

---

## Next

1. **Optuna pass** — running. Trial budgets are weighted per family (`TRIAL_BUDGET`),
   since cost per fit differs by more than 10× and EBM is both slowest and not the
   accuracy leader. Measured cost is far higher than estimated: ~19 min for LightGBM
   but ~50 min for CatBoost per variant, ~5.7 h for the full pass.

   Interim CV scores are tightly clustered — cold-start 0.367–0.373, update
   0.261–0.266, all three families within 0.006 — reinforcing that the families are
   interchangeable and the data is the binding constraint. These are training-window
   CV numbers and are **not** comparable to the held-out test-fold figures above; only
   the rolling-origin re-evaluation settles whether tuning helped.

   **Tuning is now checkpointed after every family.** It previously wrote
   `p3_best_params.json` only once the entire loop finished, so an interrupted run
   threw away hours of completed work. A restart now skips whatever is already on
   disk.
2. **Model card**, covering the blind spots P4 and P5 surfaced: the cold-start model
   cannot see off-field events (Greenwood), reputation, or injury; its residuals only
   mean something within a price stratum; and the backtest measures convergence toward
   the model, not toward realised fees.
3. **Optional — a richer mean-reversion control.** The backtest uses one lag of
   Transfermarkt's own move. An autoregressive control over the full valuation history
   would tighten the causal claim.
4. **Optional — realised-fee arbitration.** On transfers with a disclosed fee, compare
   model error against Transfermarkt error relative to what a club actually paid. Small
   n and heavy selection bias, so suggestive rather than decisive.
