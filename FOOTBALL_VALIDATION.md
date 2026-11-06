# Live-data validation, 5 November 2026

Data: football-data.org, season start years 2024–2026. The token was provided at runtime and is not included in this repository. Historical results and fixtures were downloaded successfully for all eight competitions.

Eight automated tests passed. Both CLI model selections ran successfully against the cached API responses.

## Chronological accuracy comparison

| Window start (UTC) | Competition | Matches | Previous | Enhanced | Majority baseline |
|---|---|---:|---:|---:|---:|
| 2025-09-30 | PL | 108 | 56.5% | 57.4% | 49.1% |
| 2025-09-30 | CL | 88 | 52.3% | 58.0% | 47.7% |
| 2025-09-30 | ALL | 196 | 54.6% | 57.7% | 48.5% |
| 2025-12-21 | PL | 133 | 39.8% | 38.3% | 33.8% |
| 2025-12-21 | CL | 61 | 57.4% | 50.8% | 54.1% |
| 2025-12-21 | ALL | 194 | 45.4% | 42.3% | 40.2% |
| 2026-03-17 | PL | 169 | 45.0% | 47.9% | 43.2% |
| 2026-03-17 | CL | 92 | 70.7% | 62.0% | 53.3% |
| 2026-03-17 | ALL | 261 | 54.0% | 52.9% | 46.7% |

## Final-window probability quality

Lower log loss is better.

| Competition | Previous log loss | Enhanced log loss |
|---|---:|---:|
| PL | 1.0288 | 1.0377 |
| CL | 0.8465 | 0.9621 |
| ALL | 0.9645 | 1.0111 |

The enhanced version improved PL top-choice accuracy on the final window but worsened combined accuracy and log loss, and did not improve consistently across earlier windows. The previous version therefore remains the default. This deployment decision uses the comparison above; these windows must not be presented as independent final validation for future tuning.

The final window contains 261 matches: 169 PL and 92 CL. The initial fit uses 1,039 earlier target matches. Domestic games update the enhanced state but are not extra target labels.

## Forecast smoke checks

- previous: 58 fixtures in the next 30 days; probability totals and competition filters passed; 10 low-history fixtures.
- enhanced: 58 fixtures in the next 30 days; probability totals and competition filters passed; 5 low-history fixtures.

Snapshots are not guarantees about future performance. xG, injuries, lineups and domestic cups are not included. Predictions cover regulation-time outcomes only. Raw cached API data, generated predictions and credentials are excluded from git.

## Nested selection implementation, 6 November 2026

Added a separate optional `tuned` model with inner chronological tuning, baseline blending, and temperature calibration. Outer splits now keep whole UTC match days together. Reports include Brier score, reliability bins, expected calibration error and accuracy intervals. Rest/congestion features and optional timestamp-safe xG/ClubElo imports are available to advanced candidates. The original model remains the default.

Sixteen automated tests passed, including coefficient isolation from calibration labels, outer-label isolation, same-day/future-result isolation, delayed xG availability, historical rating availability/expiry, maintenance-page rejection, PL/CL forecast probabilities and limited-history flags. GitHub Actions runs these tests on Python 3.11 and 3.12; only Python 3.12 was run locally.

### Independent public-CSV execution check

Source: football-data.co.uk E0 archives for 2023/24, 2024/25, 2025/26 and 2026/27, retrieved 6 November 2026. There were 1,230 completed Premier League results: 380 in each completed season and 90 in the current season. These are separate source IDs, not football-data.org IDs; datasets were not mixed. Time fields were interpreted in Europe/London. Missing kickoff time would use noon (the CSV importer documents this approximation). No shots, odds or other post-match columns were included in predictors. Raw data is excluded from git.

| Outer test start (UTC day) | Matches | Previous accuracy | Enhanced accuracy | Tuned accuracy | Previous log loss | Tuned log loss |
|---|---:|---:|---:|---:|---:|---:|
| 2025-02-02 | 183 | 54.1% | 56.3% | 51.9% | 1.0053 | 1.0083 |
| 2025-09-14 | 182 | 51.1% | 51.6% | 49.5% | 1.0044 | 1.0191 |
| 2026-01-24 | 250 | 45.6% | 44.8% | 44.4% | 1.0408 | 1.0381 |

The new selection procedure did **not** improve top-choice accuracy in this check. Final-window Brier score improved slightly from 0.6259 to 0.6240. This is insufficient evidence to change the default. No hyperparameters or candidate grid were changed after seeing these results. The public-CSV dataset contains no Champions League or other domestic leagues, so it does not validate cross-league benefits. These figures must not be compared directly with the different football-data.org sample above.

The completed-results archive contains no upcoming fixtures; an empty forecast CSV is expected for that execution. Upcoming fixture export was verified in synthetic PL/CL tests. To reproduce with local E0 CSV files:

```bash
python -m modern_football.public_csv path/to/E0_2324.csv path/to/E0_2425.csv path/to/E0_2526.csv path/to/E0_2627.csv --output data/football/public-pl.json
python -m modern_football --model tuned --input-json data/football/public-pl.json --output outputs/public-pl.csv
```

### Remaining live-data checks

Authenticated football-data.org execution was blocked by automatic approval review when attempting to reuse the previously supplied token. No claim is made that the revised pipeline has been run on a fresh authenticated PL/CL dataset. Normal local execution still reads `FOOTBALL_DATA_TOKEN` from the environment.

Understat returned HTTP 200 with a maintenance page and no result payload. ClubElo's historical CSV endpoint returned HTTP 503 from this environment. External import behavior is tested, but real-feed xG/ClubElo model accuracy remains unmeasured. The raw Understat downloader fails explicitly on missing results; normalized CSV imports require explicit provider IDs and point-in-time availability timestamps.
