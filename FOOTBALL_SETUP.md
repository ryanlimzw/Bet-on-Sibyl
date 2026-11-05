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

`--season` is the season's start year: 2026 means 2026/27. The default infers the season with a July boundary; use the explicit flag during summer transitions. `--history 1` uses only the selected season. Older seasons depend on your provider subscription; the program reports denied access rather than silently omitting history. Early in a season there may be too few completed games: request more history if your plan allows it.

Always use `--refresh` for current results and schedules. Without it, cached responses in `data/football` are reused, including potentially outdated results. Rate limits are reported and successful responses are cached, so retry after a minute. Requests are spaced to stay below ten per minute. Access and coverage are controlled by the provider and may change.

For offline use, `python -m modern_football --input-json path/to/matches.json` accepts a football-data v4 object with a `matches` list. Include `id`, `competition.code` (`PL` or `CL`), `utcDate` (timezone-aware ISO datetime), `status`, `stage`, `homeTeam`/`awayTeam` (`id`, `name`), and `score`. Finished games need `duration: REGULAR` plus `fullTime.home/away`, or `regularTime.home/away`. Use numeric provider team IDs consistently. Fixture statuses must be `TIMED` or `SCHEDULED`. Offline data still needs real historical results; tests use synthetic fixtures only to verify behavior.

## Model and limitations

The model learns three outcomes using logistic regression, with pre-match Elo differences, last-five match points and goal differences, history counts, competition, and final-stage indicators. Elo and form are maintained separately within each competition. No current-season aggregate table is used, avoiding the original approach's risk of future information appearing in historical features. Simultaneous kickoffs are processed together before ratings update.

The first 80% of completed games train a holdout model; the last 20% evaluate it in chronological order. Form updates after each completed kickoff in that evaluation. Then all available past results train the fixture model. Pending fixtures do not update ratings. Metrics are pooled across competitions and do not establish performance for either competition separately. No real-world accuracy claim is made until you evaluate real data.

CL targets are results after 90 minutes including stoppage time, excluding extra time/penalties. Missing regulation-time scores are skipped rather than using shootout totals. This does not predict who qualifies, aggregate two-leg winners, or exact scores. Finals have a neutral-venue Elo update, but the classifier is only a basic venue approximation. Low-history teams start at equal Elo and are flagged; CL-only history omits most clubs' domestic form. Injuries, lineups, odds, rest, and transfers are not modeled. Compare holdout performance against the baseline before relying on predictions.

Provider documentation: https://www.football-data.org/documentation/quickstart and https://docs.football-data.org/general/v4/overtime.html

Validation: automated tests cover probability totals, chronological feature isolation, simultaneous fixtures, regulation-time scores, cache/auth errors, empty fixture windows, and inadequate history. Live authenticated retrieval requires your own token and was not executed during development.
