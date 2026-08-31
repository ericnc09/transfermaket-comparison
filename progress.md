# Progress log

Living record of this project: the plan, every decision and why, and what has
actually been built. **Updated at the start of every working session and before
every commit or push.**

Last updated: 2026-08-30 · review complete; P6 planned and ready to execute

---

## Status

| Phase | Scope | State |
|---|---|---|
| **P0** | Kill-risk spike — prove the FBref ↔ Transfermarkt join | ✅ Complete |
| **P1** | Data platform — feature matrix, panel, leakage gates | ✅ Complete |
| **P2** | First models — baselines + GBM, temporal eval | ✅ Complete |
| **P3** | Model zoo — full tier list, quantiles, ensemble, tuning | ✅ Complete |
| **P4** | Explain & compare — SHAP, similarity, residual leaderboard | ✅ Complete |
| **P5** | Research — forward backtest, model card | ✅ Complete |
| **P6** | Extended panel, re-tune, full zoo, fingerprint | ▶ Planned — see [Next](#next--p6-extended-panel-re-tune-full-zoo) |

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

> **Superseded and regenerated.** The table originally published here was produced
> before the P4 corrections (deflated target, `Season_End_Year`/`Born` as features)
> and was stale. These are the numbers the current code produces. Only the key rows
> were re-run; the full zoo (HistGBM, XGBoost, EBM, Ridge, Stacked) still needs a
> pass and is not quoted until it has one.

| Model | log MAE | ±sd | log R² | MedAE €m | ±30% | NDCG@100 |
|---|---|---|---|---|---|---|
| **CatBoost [update]** | **0.261** | 0.033 | 0.922 | 1.28 | 66.5% | 0.934 |
| LightGBM [update] | 0.271 | 0.041 | 0.917 | 1.37 | 65.3% | 0.946 |
| **CatBoost [coldstart]** | **0.364** | 0.059 | 0.853 | 1.79 | 52.6% | 0.894 |
| LightGBM [coldstart] | 0.372 | 0.050 | 0.847 | 1.89 | 50.7% | 0.905 |
| *carry forward prior TM value* | *0.497* | *0.051* | *0.641* | *2.33* | *42.4%* | *0.910* |
| log-linear (age, minutes, G+A) | 0.725 | 0.014 | 0.463 | 3.67 | 25.3% | 0.712 |
| median by position × age | 0.883 | 0.033 | 0.201 | 4.75 | 20.6% | 0.281 |
| global median | 1.010 | 0.015 | −0.017 | 4.60 | 15.9% | 0.146 |

Every P4 correction improved the model: cold-start 0.401 → **0.364**, update
0.285 → **0.261**. The cold-start margin over carry-forward widened from 19% to
**27%**. Note `within_30pct` also changed definition in the review (see below), so
that column is not comparable to the old table.

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

## ML review — findings and fixes

A full pass over the maths and parameters: scoring functions, target transform,
conformal calibration, stacking, splits, feature manifest, and the backtest
specification.

### Critical: every published leaderboard number was stale

The P3 leaderboard was written 2026-08-28 19:11. `gbm.py` (target), `manifest.py`
(features) and `build.py` (deflator) were all modified 2026-08-30 13:46–13:56. The
numbers quoted in this file, the model card, the design artifact and the draft posts
were therefore produced by a configuration that no longer existed. **The leaderboard
should have been regenerated when the target changed.** Corrected above.

### Bugs found and fixed

| Bug | Effect | Fix |
|---|---|---|
| `Stacked` re-instantiated bases as `proto.__class__(proto.variant)` | Silently discarded every tuned hyperparameter — the stacker's bases were always defaults, even in the tuned run | `copy.deepcopy(proto)` |
| Stale `TRAIN`/`VALID`/`TEST` constants hardcoding 2022 as excluded | P3 repaired that season; anything using `temporal_split` silently dropped a fifth of the panel | Constants and function removed; `rolling_origin` is the protocol |
| `within_30pct` band was `(0.7, 1.3)` | Asymmetric in log space: counted a 30% under-prediction, forgave a 43% over-prediction | `(1/1.3, 1.3)` |
| Conformal fell back to in-sample calibration when the slice was small | Would produce intervals that look tight and cover nothing | Raises instead |

### The headline backtest survived a harder test

A **placebo** was constructed: a "model" with zero football knowledge that predicts
each player's price-stratum mean, so its residual is purely the negative of his
deviation from that mean — pure mean reversion.

| | β | p |
|---|---|---|
| Placebo alone | +0.0765 | 0.032 |
| Real residual alone | +0.1871 | 2.8e-25 |
| **Both — real** | **+0.1855** | **7.4e-25** |
| **Both — placebo** | +0.0355 | 0.30 |

The confound is real on its own, but the two are near-orthogonal (r=0.098): with both
in the regression the real residual is unchanged and the placebo collapses. **Mean
reversion is not what drives the finding.** This supersedes the low-momentum subsample
as the primary robustness check — it is cleaner, since conditioning on low momentum is
itself conditioning on a function of the noise.

### Caveats now on record

- **Backtest attrition is not random.** 29.5% of rows have no forward value, and they
  differ systematically: mean price-adjusted residual −0.049 dropped vs +0.021 retained
  (t=5.83, p=5.9e-09). Players the model marked down disproportionately vanish from
  coverage. β is conditional on surviving in the market.
- **The first rolling-origin fold trains on one season** (1,701 rows) and scores 0.536
  against 0.325 for the last, dragging the mean from 0.356 to 0.401 in the old table.
  Averaging over unequal training sizes should be stated, not silent.
- **Conformal predicts from a base fit on 75% of train**, so P4's point predictions come
  from a weaker model than the leaderboard's. The two are not directly comparable.
- **`p10`/`p90` are convenience names** for a symmetric fixed-width conformal band, not
  percentile estimates.
- **Conformal coverage sits just below nominal** (0.79 vs 0.80) because a temporal split
  breaks the exchangeability its guarantee assumes.

Clean on inspection: per-90 derivation, NDCG, R², the log/expm1 round-trip, grouped CV,
and no feature correlating above 0.9 with the target.

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

## Next — P6: extended panel, re-tune, full zoo

**This is the plan to execute. It is written to be picked up cold in a new session.**
Read this section, then start at Phase 1.

### Why this exists

The ML review found that every published leaderboard number was stale, and fixing that
surfaced two further problems worth solving properly rather than patching:

- **The tuned hyperparameters are stale too.** `p3_best_params.json` is dated
  2026-08-28 18:37; `gbm.py` and `manifest.py` changed 2026-08-30. They were optimised
  against the deflated target with `Season_End_Year`/`Born` as features. **Decision: re-tune.**
- **The first rolling-origin fold trains on a single season** (1,701 rows) and scores
  0.536 against 0.325 for the last, dragging the mean from 0.356 to 0.401. **Decision:
  report every fold, and extend the panel backwards so no fold is that badly trained.**

### The opportunity

The panel starts at Season_End_Year 2018 only because that is when FBref's *advanced*
tables begin. The basic tables go back much further:

| Table | Seasons available |
|---|---|
| FBref standard / shooting / misc / playing_time | **2010 → 2023** |
| FBref advanced (passing, defense, possession, gca, passing_types) | 2018 → 2023 |
| Transfermarkt mirror labels (`tm_vals.parquet`) | 2010 → 2022 |

Extending training to Season_End_Year 2012 adds **~10,300 rows** (panel ~18,600), and
the fold testing 2019 would train on **seven seasons instead of one**.

### Phase 1 — build the extended panel and A/B it (~1 hour)

Do **not** skip to Phase 2. Tuning takes most of a working day and must not run against
a panel that turns out to be worse.

1. Widen `SEASONS` in `src/features/build.py` from `range(2018, 2024)` to
   `range(2012, 2024)`. Advanced-table columns will be NaN before 2018; the tree models
   handle that natively and `feature_columns(..., require_variance=True)` already drops
   anything degenerate.
2. Re-run entity resolution over the wider window — the resolver is unchanged and took
   11 s for six seasons, so this is cheap. Regenerate `data/interim/big5_resolution.parquet`.
3. `python scripts/03_build_panel.py`. Expect the existing gates to fire on anything
   unexpected; `stale_label_seasons()` and `partial_seasons()` are data-driven and should
   still flag only 2022 (mirror labels) and 2023 (partial) respectively.
4. **A/B head to head**, LightGBM + CatBoost, both variants, on **identical test folds
   (2019–2022)**: 5-season panel vs extended. Only the training window differs.

**Two risks that must be checked at the decision point, not assumed away:**

- **Missingness as a season proxy.** Advanced features are NaN if and only if the season
  is pre-2018, and older seasons have systematically lower values. That is precisely the
  leak `Season_End_Year` was removed to prevent, re-entering through the NaN pattern.
  Test rows always have full features so the missing branch never fires at inference, but
  training on mixed completeness can still distort what is learned for complete rows.
  Check: does per-fold performance on 2019–2022 actually improve, and does SHAP show the
  model keying on advanced-feature presence?
- **Mixed label provenance.** 2018+ labels are precisely dated scraped valuations;
  pre-2018 would be mirror season-start snapshots. Acceptable for training-only seasons,
  but it is an inconsistency to state, not hide.

**Decision point: keep the extended panel only if it measurably improves held-out
performance on the common folds.** If it does not, revert `SEASONS` and proceed with the
five-season panel. Record the outcome here either way.

### Phase 2 — the full run (~7–9 hours, checkpointed)

5. **Re-tune** on the winning panel: `python scripts/06_tune_and_evaluate.py --trials 40`.
   Per-family budgets live in `TRIAL_BUDGET`. It checkpoints `p3_best_params.json` after
   every family and resumes, so it can run overnight and survive interruption. Delete the
   old `p3_best_params.json` first — it is stale and would be silently reused.
   Expect longer than the previous 5.7 h on a bigger panel.
6. **Full zoo**: 4 baselines, Ridge/ElasticNet+splines, LightGBM, XGBoost, CatBoost,
   HistGBM, EBM, and the **full Stacked** (LightGBM + CatBoost + EBM). The stacker now
   deep-copies its bases, so it finally honours tuned parameters.
7. **Seed-stability check**: top two models × 3 seeds on one fold, ~2 min. Fold-to-fold
   sd is 0.05–0.06; if seed variance is comparable, the gaps between families are noise
   and the write-up must say so instead of ranking them.
8. **Config fingerprint.** Write git SHA + a hash of the feature list + the target column
   name into every results JSON, and add a test that fails when a published leaderboard's
   fingerprint does not match the current code. This is the specific failure that cost us
   a full set of wrong numbers; make it structurally impossible to repeat.

### After the run — propagate

Numbers appear in four places and **all four go stale together**:

- `progress.md` (this file)
- `MODEL_CARD.md`
- the design artifact — <https://claude.ai/code/artifact/7f2391b1-149f-4f3b-bbf0-37063d6d38dd>
- the write-up pitch, whose draft LinkedIn copy quotes the leaderboard —
  <https://claude.ai/code/artifact/fa9fc39a-0258-44f1-b25a-9abd506a0872>

**Do not publish anything from the pitch artifact until it is refreshed.** Its current
copy cites 0.401 vs 0.497; the corrected figures are 0.364 vs 0.497, which makes the
margin 27% rather than 19% — a stronger claim, not a weaker one.

### Current reference numbers (post-review, current code)

Key rows only. HistGBM, XGBoost, EBM, Ridge and Stacked have **not** been re-run since
the P4 corrections and must not be quoted until they have been.

| Model | log MAE | ±sd | log R² | MedAE €m | ±30% |
|---|---|---|---|---|---|
| CatBoost [update] | 0.261 | 0.033 | 0.922 | 1.28 | 66.5% |
| CatBoost [coldstart] | 0.364 | 0.059 | 0.853 | 1.79 | 52.6% |
| *carry forward prior TM value* | *0.497* | *0.051* | *0.641* | *2.33* | *42.4%* |

Regenerate with `python scripts/10_recheck_leaderboard.py`.

---

## Optional, after P6

1. **A richer mean-reversion control.** The backtest uses one lag of Transfermarkt's own
   move. An autoregressive control over the full valuation history would tighten the
   causal claim beyond what the placebo test already establishes.
2. **Realised-fee arbitration.** On transfers with a disclosed fee, compare model error
   against Transfermarkt error relative to what a club actually paid. The only test with
   ground truth outside Transfermarkt, and the missing leg for a paper. Small n and heavy
   selection bias, so suggestive rather than decisive. Fee data is already in the mirror.
3. **The write-up.** Three angles drafted in the pitch artifact above; recommended
   sequence is the debugging story on LinkedIn, the market-efficiency piece on Substack.
