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
