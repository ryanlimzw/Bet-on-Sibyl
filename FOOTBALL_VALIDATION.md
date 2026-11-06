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

Sixteen automated tests passed, including coefficient isolation from calibration labels, outer-label isolation, same-day/future-result isolation, delayed xG availability, historical rating availability/expiry, maintenance-page rejection, PL/CL forecast probabilities and limited-history flags. GitHub Actions passed these tests on Python 3.11 and 3.12; Python 3.12 was also run locally.

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

### External-feed checks

Authenticated football-data.org validation was completed after explicit authorization to reuse the token, as recorded below. The initial automatic credential-transfer review block is resolved. Normal local execution reads `FOOTBALL_DATA_TOKEN` from the environment; no credentials are saved in git.

Understat returned HTTP 200 with a maintenance page and no result payload. ClubElo's historical CSV endpoint returned HTTP 503 from this environment. External import behavior is tested, but real-feed xG/ClubElo model accuracy remains unmeasured. The raw Understat downloader fails explicitly on missing results; normalized CSV imports require explicit provider IDs and point-in-time availability timestamps.

## Authenticated PL/CL validation, 6 November 2026

Downloaded all 24 football-data.org league-season responses for season start years 2024–2026. The 7,614 unique matches include 5,816 completed results across eight competitions. Training targets comprise 850 PL and 450 CL games (1,300 total); other domestic results update enhanced/advanced features. No completed PL/CL match was skipped for a missing regulation-time score. The token was supplied only in the process environment and is not stored in source or cached responses.

Forecast snapshot: `2026-11-06T04:01:36.760422+00:00`. Completed-input SHA256: `3115506ae949ea3c59e6621fa0d435f13bbadab27e6c579d726f11d966d67a88`.

Each model was run through the CLI using `--season 2026 --history 3 --days 30 --model NAME` against the freshly downloaded cache. All three reports used identical completed-data fingerprints and evaluation folds. Checks confirmed finite probabilities in [0,1], sums of one, no duplicate exported fixtures, PL/CL-only output, and exact agreement with eligible provider fixtures in the forecast window.

| Outer test start (UTC day) | Competition | Matches | Previous accuracy | Enhanced accuracy | Tuned accuracy |
|---|---|---:|---:|---:|---:|
| 2025-09-30 | PL | 108 | 55.6% | 55.6% | 57.4% |
| 2025-09-30 | CL | 90 | 53.3% | 58.9% | 42.2% |
| 2025-09-30 | ALL | 198 | 54.5% | 57.1% | 50.5% |
| 2025-12-21 | PL | 133 | 39.8% | 38.3% | 38.3% |
| 2025-12-21 | CL | 60 | 56.7% | 50.0% | 50.0% |
| 2025-12-21 | ALL | 193 | 45.1% | 42.0% | 42.0% |
| 2026-03-17 | PL | 169 | 45.0% | 47.9% | 46.7% |
| 2026-03-17 | CL | 93 | 71.0% | 62.4% | 63.4% |
| 2026-03-17 | ALL | 262 | 54.2% | 53.1% | 52.7% |

### Final-window probability quality

| Competition | Previous log loss | Enhanced log loss | Tuned log loss | Previous Brier | Tuned Brier |
|---|---:|---:|---:|---:|---:|
| PL | 1.0288 | 1.0377 | 1.0456 | 0.6164 | 0.6280 |
| CL | 0.8462 | 0.9549 | 0.9457 | 0.4700 | 0.5572 |
| ALL | 0.9640 | 1.0083 | 1.0101 | 0.5644 | 0.6028 |

The optional tuned model does not improve consistently. On the final 262 matches, accuracy is 52.7% versus 54.2% for the previous model, and log loss worsens from 0.9640 to 1.0101. It improves final-window PL top-choice accuracy from 45.0% to 46.7%, but CL accuracy falls from 71.0% to 63.4%. Across all three disjoint outer periods (653 games), tuned accuracy is 48.9% versus 51.6%, with log loss 1.0286 versus 1.0035. The previous model remains the default; no candidate grid, hyperparameters or features were changed in response to this run.

The tuned production selection chose the previous-feature model with temperature 1.0. Its coefficients use the first 1,038 targets; the remaining 262 form its calibration suffix. The default previous model still fits all available targets. These are distinct fits even when they share a feature specification.

Whole-day split boundaries move several matches compared with the earlier timestamp-split report: outer test counts are now 198, 193 and 262. The periods were previously inspected, so these results remain development evidence rather than a new untouched future validation.

### Upcoming fixture smoke checks

| Exported model | Fixtures | PL | CL | Limited-history fixtures |
|---|---:|---:|---:|---:|
| previous | 65 | 47 | 18 | 10 |
| enhanced | 65 | 47 | 18 | 5 |
| tuned | 65 | 47 | 18 | 10 |

Upcoming fixture CSVs and full diagnostic JSON reports were generated locally and remain excluded from git, along with the raw API cache. Fresh PL/CL execution is complete. Real xG/ClubElo ingestion and its predictive benefit remain unvalidated because those feeds were unavailable; the optional import paths retain synthetic coverage.
