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
