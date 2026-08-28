# transfermaket-comparison

Machine-learning transfer value predictor for Big 5 outfield footballers — trained to
reproduce Transfermarkt market values, and to surface where it disagrees with them.

**Status: P1 (data platform) complete.** Panel built and gated; no model trained yet.

## P1 result — the modelling panel

| | |
|---|---|
| Panel rows (eligible + labelled) | **7,077** |
| Unique players | **2,688** |
| Features | 177 cold-start / 180 update |
| Seasons | 2017-18 → 2021-22 (five label-able) |
| Leagues | Big 5, 1,300–1,480 rows each |
| Store | `data/processed/market.duckdb` + parquet |

Alignment is the one the model uses: **stats from season *t* predict the Transfermarkt
valuation published at the start of season *t+1*.** Nothing dated on or after the label
enters the feature matrix.

Built by `scripts/03_build_panel.py`:

- Nine FBref stat tables merged on `(season, squad, competition, player_url)` into one
  wide player-season matrix.
- **Mid-season transfers collapsed** — 638 player-seasons spanned two clubs. Counting
  stats are summed so a January mover keeps his whole season; rate-like columns take a
  minutes-weighted mean; context comes from the club he played most minutes for.
- 61 counting stats converted to per-90 rates, plus ratio features (shot accuracy,
  aerial win %, dribble success), overperformance (`G−xG`, `A−xAG`) and `npxG+xAG/90`.
- Transfermarkt supplies label, prior valuation, date of birth, height, foot,
  contract expiry and date joined. Age and contract-months-left are computed
  **relative to the label date**, not the season.
- Context: squad total market value and its rank within the league, league median
  value, and target deflation by that median so the model learns quality, not inflation.
- True season-1 and season-2 lags, joined explicitly on `(player, season−k)` so a
  missing season yields null rather than a silently shifted row.

151 of 180 features are ≥99% populated. The 11 below 50% are all `lag2` columns, which
are structurally absent for a player's first two observed seasons.

### Leakage gates

`tests/test_panel.py` holds the guarantees, all enforced on every run:

- No feature correlates >0.98 with the target.
- `prior_value` is verified against the raw source to come from season *t−1* and the
  label from season *t+1* — checked row by row on a random sample.
- Lag columns are verified to equal the same player's value one season earlier.
- The cold-start variant provably excludes every prior-valuation column.
- No target column appears in either feature set.

## P0 result

The phase existed to answer one question: can FBref player-seasons be joined to
Transfermarkt valuations reliably enough to build on? Yes.

| Population | Match rate |
|---|---|
| Big 5, outfield, ≥600 min, 2017-18 → 2022-23 | **98.8%** (9,376 / 9,492) |
| Premier League only | 99.1% |
| Worst league (La Liga) | 97.7% |
| Worst single league-season | 94.1% (La Liga 2022-23, truncated source data) |

Signal is present and correctly signed on the Premier League panel (n=1,550):

| Feature | Spearman ρ vs value |
|---|---|
| age | **−0.467** |
| npxG + xAG per 90 | +0.319 |
| minutes played | +0.301 |

Median value rose €12.0m → €20.0m across five seasons, confirming that market
inflation has to be deflated out before modelling.

## Data sources

FBref's own site sits behind a Cloudflare JS challenge, so stats come from the
[worldfootballR_data](https://github.com/JaseZiv/worldfootballR_data) mirror, which
publishes the same Opta-derived season tables alongside Transfermarkt squad
valuations. Both are cached locally as parquet on first use.

**Coverage ceiling: 2017-18 → 2022-23** (6 seasons). The mirror stopped updating in
late 2022. Transfermarkt values are one snapshot per season, not the ~3/year the
design assumes.

## Entity resolution

No shared key exists between sources. `src/entity/resolve.py` blocks on
(season, competition) and runs a confidence cascade:

| Tier | Rule | Share |
|---|---|---|
| A | exact normalised name + birth year | 94.3% |
| B | exact name, unique in block | 0.1% |
| C | birth year + fuzzy name ≥ 80 | 4.3% |
| D | same club + fuzzy name ≥ 85 | <0.1% |
| E | fuzzy ≥ 92, unambiguous | <0.1% |

Tier C is what catches word-order and nickname variants — *Son Heung-min* vs
*Heung-min Son*, *Pierre Højbjerg* vs *Pierre-Emile Højbjerg*, *Martinelli* vs
*Gabriel Martinelli*.

## Usage

```bash
python -m venv .venv && ./.venv/bin/pip install -r requirements.txt
./.venv/bin/python scripts/01_ingest.py      # download + cache sources
./.venv/bin/python scripts/02_p0_panel.py    # P0: resolve + signal check
./.venv/bin/python scripts/03_p0_report.py   # P0: spot-check + figure
./.venv/bin/python scripts/03_build_panel.py # P1: full panel -> duckdb
./.venv/bin/python -m pytest tests/ -q       # match-rate + leakage gates
```

## Layout

```
src/ingest/sources.py        source download + parquet cache
src/entity/resolve.py        the join (name normalisation, club crosswalk, cascade)
src/features/definitions.py  declarative feature spec (61 counting stats)
src/features/build.py        merge, aggregate, rate, attach, lag, deflate
src/features/manifest.py     which columns are features, per variant
scripts/                     numbered pipeline stages
tests/                       match-rate gate, leakage gates, unit tests
data/{raw,interim,processed}/
```
