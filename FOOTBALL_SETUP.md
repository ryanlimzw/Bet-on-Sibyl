# Premier League and UEFA Champions League (Python 3)

Use the new `modern_football` module. The original Python 2 scripts are retained as historical reference and are not part of this workflow. This is a new baseline model, not a validated reproduction of the original algorithm.

## Windows setup

Install Python 3.11 or 3.12 from https://www.python.org/downloads/ and Git from https://git-scm.com/downloads. Open PowerShell:

```powershell
git clone https://github.com/ryanlimzw/Bet-on-Sibyl.git
cd Bet-on-Sibyl
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-football.txt
```

Register at https://www.football-data.org/client/register to get your own API token. Set it locally, never commit it:

```powershell
$env:FOOTBALL_DATA_TOKEN = "YOUR_TOKEN"
.\.venv\Scripts\python.exe -m modern_football --refresh
```

The token must be set again in each new terminal. Python 3.11 users can replace `-3.12` with `-3.11`. No browser driver, GPU, Anaconda, or PyCharm is required.

## macOS / Linux

```bash
git clone https://github.com/ryanlimzw/Bet-on-Sibyl.git
cd Bet-on-Sibyl
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-football.txt
export FOOTBALL_DATA_TOKEN="YOUR_TOKEN"
.venv/bin/python -m modern_football --refresh
```

A phone is not the intended local runtime. Run this on a computer; a hosted notebook would need separate setup.

## Results and options

`outputs/predictions.csv` contains PL and CL fixtures over the next 14 days, kickoff times in UTC, home/draw/away probabilities, predicted outcome (`H`, `D`, `A`), and history coverage flags. Open it in Excel. `outputs/predictions.metrics.json` contains chronological holdout accuracy, majority baseline accuracy, and log loss. Probabilities sum to one; they are not guaranteed outcomes or calibrated betting confidence.

```bash
python -m modern_football --season 2026 --history 3 --days 30 --refresh
python -m unittest discover -s tests -v
```

`--season` is the season's start year: 2026 means 2026/27. The default infers the season with a July boundary; use the explicit flag during summer transitions. `--history 3` is now the default; `--history 1` uses only the selected season. Older seasons depend on your provider subscription; the program reports denied access rather than silently omitting history. Early in a season there may be too few completed games: request more history if your plan allows it.

Always use `--refresh` for current results and schedules. Without it, cached responses in `data/football` are reused, including potentially outdated results. Rate limits are reported and successful responses are cached, so retry after a minute. Requests are spaced to stay below ten per minute. Access and coverage are controlled by the provider and may change.

For offline use, `python -m modern_football --input-json path/to/matches.json` accepts a football-data v4 object with a `matches` list. Include `id`, `competition.code` (`PL` or `CL`), `utcDate` (timezone-aware ISO datetime), `status`, `stage`, `homeTeam`/`awayTeam` (`id`, `name`), and `score`. Finished games need `duration: REGULAR` plus `fullTime.home/away`, or `regularTime.home/away`. Use numeric provider team IDs consistently. Fixture statuses must be `TIMED` or `SCHEDULED`. Offline data still needs real historical results; tests use synthetic fixtures only to verify behavior.

## Selecting the prediction model

The previous model remains the default because the enhanced features did not improve consistently in chronological testing. Both are evaluated on every run. To explicitly use domestic history and weighted attack/defence features:

```bash
python -m modern_football --model enhanced --history 3 --days 30 --refresh
```

Use `--model previous` for the default model. Data collection includes all eight competitions for comparison even when exporting previous-model predictions. The `model` field in the metrics JSON identifies the model used for the CSV. Choosing a default after this comparison is a deployment decision; the reported windows are now development evidence, not a fresh untouched test of future changes.

## Model and limitations

The enhanced model uses domestic results from England, Germany, Spain, Italy, France, Netherlands, and Portugal, plus Champions League results. Only Premier League and Champions League matches become training targets or exported predictions. A shared team Elo rating connects domestic and European performances by stable provider team ID. Cross-league comparisons are learned indirectly through European games; there is no externally validated league-strength ranking.

Features include shared Elo differences, separate goals scored and conceded rates, points form, home/away venue form, history counts, competition, and final-stage flags. Form uses up to 30 matches (20 per venue), a 120-day exponential half-life and a five-match prior to soften sparse histories. Training examples receive a 365-day half-life. These are fixed starting settings, not parameters tuned on the reported test sets. Older results still affect Elo; only form and training samples decay explicitly.

All features use only completed results from prior UTC calendar days during historical feature construction. Same-day games update state together after feature construction, preventing results from overlapping games leaking into features. For upcoming fixtures the state uses all currently completed historical games. Future scheduled matches never update ratings or form.

Metrics compare the previous and enhanced model on identical PL/CL target matches. Three expanding windows cover 50–65%, 65–80%, and 80–100% of chronological history; the last is the final holdout. Each window reports accuracy and log loss separately for PL and CL and combined, alongside a training-majority baseline. Holdout results update form as time advances, while model coefficients remain fixed within each window. No hyperparameter search or model selection uses these results. The previous baseline is reconstructed with the same prior-day safety rule, so results may differ slightly from the earlier version.

The output CSV retains home/draw/away probabilities and history flags. Enhanced-model counts include all covered competitions; previous-model counts remain competition-specific. The JSON report contains a `folds` list with `previous` and `enhanced` model results. More features need not improve accuracy; inspect both accuracy and log loss rather than assuming this version is better.

CL targets are results after 90 minutes including stoppage time, excluding extra time/penalties. Missing regulation-time scores are skipped. This does not predict who qualifies, aggregate two-leg winners, or exact scores. Finals omit home advantage in the Elo update and venue-form updates, but neutral venues are only approximated. Sparse-history teams start at equal Elo and are flagged.

This upgrade does not add xG, injuries, lineups, shots, odds, or rest-day features. Domestic cups and leagues outside the listed coverage remain missing. Injuries and transfers are not modeled. The API token remains an environment variable and is never stored in the repository. Every requested league/season must be accessible; denied access produces an error rather than silently substituting missing data.

Provider documentation: https://www.football-data.org/documentation/quickstart and https://docs.football-data.org/general/v4/overtime.html

Validation includes chronological feature isolation, domestic-to-European information flow, time decay, venue tracking, smoothing, regulation-time scores, API cache/auth errors, inadequate history, and probability totals. See `FOOTBALL_VALIDATION.md` for the measured live-data comparison.

## Optional tuned and calibrated model

```bash
python -m modern_football --model tuned --history 3 --days 30 --refresh
```

The `tuned` candidate adds rest days, matches played in the previous 14 days, and missing-history indicators. It tests 13 specifications: the original baseline plus enhanced/advanced features with logistic regularization C in {0.1, 0.5, 2} and training half-life in {180, 365} days. Form smoothing remains fixed. The best specification and optional 25/50/75% blend with the original baseline are selected by log loss on an inner tuning period. This is a small logistic-model ensemble, not a tree or neural-network ensemble.

Each outer training prefix is divided at whole UTC day boundaries: first 60% fits candidates; 60–80% selects the specification/blend; first 80% then fits its coefficients; final 20% selects a probability temperature from {0.75, 1, 1.25, 1.5, 2}. Coefficients are deliberately not refit on calibration labels. Production uses the same procedure on all available history, so coefficients exclude the most recent calibration suffix. This trades coefficient freshness for a genuinely separate calibration period. Calibration can help or hurt future accuracy; a calibrated candidate is not a claim of established calibration quality.

Outer test periods are never passed into model selection or calibration. Earlier completed test-period matches still update team form for subsequent days. Existing previously inspected periods remain development evidence. A future untouched period is required before changing the default. The original model remains the default.

Every model now reports multiclass Brier score (sum across classes, range 0–2), top-choice probability reliability bins, expected calibration error, log loss, and a Wilson interval for accuracy. Wilson intervals assume independent outcomes and can understate uncertainty for correlated team schedules. The metrics JSON records selected members, weights, dates, temperature and sample counts. Small bins and small calibration periods are noisy.

### Optional historical xG and ClubElo

```bash
python -m modern_football --model tuned --input-json data/football/matches.json --xg-csv data/football/xg.csv --clubelo-csv data/football/clubelo.csv
```

Normalized xG CSV columns:

```csv
match_id,home_team_id,away_team_id,kickoff_utc,home_xg,away_xg,available_at_utc
```

Use actual football-data.org match/team IDs and timezone-aware ISO timestamps. Joins require matching fixture ID, both team IDs and kickoff. xG values must be finite and nonnegative. Duplicate IDs are errors; unmatched records are counted as unmatched implicitly through the matched-record count. Historical availability must be after kickoff. Features only use completed prior-day matches whose xG was available before the target's UTC calendar day. Missing xG uses a smoothed prior plus coverage indicators, never the target's own xG. A current download timestamp is safe but will not establish historical availability for backtesting. Retrospectively reconstructed publication times are assumptions and must be recorded as such.

Normalized ClubElo CSV columns:

```csv
team_id,effective_at_utc,available_at_utc,elo
```

Map ClubElo names to provider team IDs explicitly before importing. The latest rating whose effective and availability dates both precede the target UTC day is used. Snapshots older than 30 days are treated as missing; coverage and age indicators remain available. Today's ratings must never be copied into historical rows. These external features are available only in advanced candidates; tuning can choose a candidate that does not use them. Neither CSV is required.

A raw Understat downloader is provided:

```bash
python -m modern_football.external --league EPL --season 2025 --output data/football/understat-EPL-2025.json
```

It rejects maintenance pages and changed/missing embedded result payloads. Its raw output still needs explicit team/fixture ID mapping and genuine availability timestamps before producing the normalized xG CSV. There is no guessed name matching. Do not pass the raw output to `--input-json`, which expects football-data.org format.

On 6 November 2026 Understat returned a maintenance page (HTTP 200 without results), and the ClubElo CSV API returned HTTP 503 in this environment. Live external-feed ingestion and its accuracy benefit could not be validated. The import paths were tested with synthetic timestamped records; no xG or ClubElo improvement is claimed.

### Local public Premier League CSV validation

`python -m modern_football.public_csv E0_2324.csv E0_2425.csv --output data/football/public-pl.json` converts local football-data.co.uk Premier League results for `--input-json`. It handles UTF-8 BOMs, deduplicates identical fixtures and rejects conflicting results. Kickoff times are interpreted in Europe/London; missing times use noon, which approximates rest intervals. IDs use a separate `fdcsv:` namespace and must not be mixed with football-data.org IDs. This importer includes only finished PL results, no CL history or future fixtures. It does not download data. See `FOOTBALL_VALIDATION.md` for the actual execution results.
