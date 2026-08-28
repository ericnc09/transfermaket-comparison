# Progress log

Living record of this project: the plan, every decision and why, and what has
actually been built. **Updated at the start of every working session and before
every commit or push.**

Last updated: 2026-08-28 · after P1 (data platform)

---

## Status

| Phase | Scope | State |
|---|---|---|
| **P0** | Kill-risk spike — prove the FBref ↔ Transfermarkt join | ✅ Complete |
| **P1** | Data platform — feature matrix, panel, leakage gates | ✅ Complete |
| **P2** | First models — baselines + LightGBM, temporal eval | ▶ In progress |
| **P3** | Model zoo — full tier list, Optuna, quantiles, ensemble | ⬜ Not started |
| **P4** | Explain & compare — SHAP, similarity, residual leaderboard | ⬜ Not started |
| **P5** | Research — forward backtest, model card, writeup | ⬜ Not started |

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
| Rows (eligible + labelled) | **7,077** |
| Unique players | **2,688** |
| Features | 177 cold-start / 180 update |
| Seasons | 2017-18 → 2021-22 (five label-able) |
| Leagues | Big 5, 1,300–1,480 rows each |
| Store | `data/processed/market.duckdb` + parquet |

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

---

## Next — P2

Baselines and the first real model.

1. Tier 0 baselines, all four, including **last season's TM value carried forward** —
   the one that must be reported alongside every headline number.
2. LightGBM on both variants.
3. Temporal split: train 2017-18 → 2019-20, validate 2020-21, test 2021-22.
4. Player-grouped CV inside the training window.
5. Metrics in log space, money space (MAE, **MedAE**, sMAPE, ±30% hit rate) and rank
   space (Spearman, NDCG@100), segmented by position, value decile, league, age band.
