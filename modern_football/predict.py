"""Chronological Elo/form features with multinomial logistic regression."""
import argparse
import json
import os
import sys
import time
from collections import defaultdict, deque
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

COMPETITIONS = ('PL', 'CL')
LABELS = ('A', 'D', 'H')


def fetch(competition, season, token, cache, refresh=False):
    path = cache / f'{competition}_{season}.json'
    if path.exists() and not refresh:
        return json.loads(path.read_text(encoding='utf-8'))
    if not token:
        raise ValueError('Set FOOTBALL_DATA_TOKEN or supply cached data. See FOOTBALL_SETUP.md.')
    response = requests.get(
        f'https://api.football-data.org/v4/competitions/{competition}/matches',
        params={'season': season}, headers={'X-Auth-Token': token}, timeout=45)
    if response.status_code in (401, 403):
        raise ValueError(f'{competition} {season}: token invalid or season unavailable on your API plan. '
                         'Try --history 1, check your subscription, or use --input-json.')
    if response.status_code == 429:
        raise ValueError('API rate limit reached. Wait a minute and retry; successful requests are cached.')
    response.raise_for_status()
    data = response.json()
    if not isinstance(data.get('matches'), list):
        raise ValueError('API response is missing matches.')
    cache.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding='utf-8')
    time.sleep(6.1)  # keep sequential requests below ten per minute
    return data


def score90(match):
    score = match.get('score', {})
    result = score.get('regularTime')
    if not result and score.get('duration') == 'REGULAR':
        result = score.get('fullTime')
    if not result or any(result.get(k) is None for k in ('home', 'away')):
        return None  # never train on shootout/extra-time totals
    return int(result['home']), int(result['away'])


class State:
    def __init__(self):
        # Separate competition ratings avoid mixing PL domestic and CL-only schedules.
        self.ratings = defaultdict(lambda: 1500.0)
        self.form = defaultdict(lambda: deque(maxlen=5))
        self.count = defaultdict(int)

    def keys(self, m):
        c = m['competition']['code']
        return (c, m['homeTeam']['id']), (c, m['awayTeam']['id'])

    def features(self, m):
        h, a = self.keys(m)
        def avg(k, col):
            return np.mean([r[col] for r in self.form[k]]) if self.form[k] else 0.0
        return [ (self.ratings[h] - self.ratings[a]) / 400,
                 avg(h, 0) - avg(a, 0), avg(h, 1) - avg(a, 1),
                 min(self.count[h], 20) / 20, min(self.count[a], 20) / 20,
                 float(h[0] == 'CL'), float(m.get('stage') == 'FINAL') ]

    def update(self, m):
        score = score90(m)
        if score is None:
            return
        hg, ag = score
        h, a = self.keys(m)
        # Finals are treated as neutral venues.
        advantage = 0 if m.get('stage') == 'FINAL' else 60
        expected = 1 / (1 + 10 ** ((self.ratings[a] - self.ratings[h] - advantage) / 400))
        actual = 1 if hg > ag else 0.5 if hg == ag else 0
        delta = 20 * (actual - expected)
        self.ratings[h] += delta
        self.ratings[a] -= delta
        self.form[h].append((3 if hg > ag else 1 if hg == ag else 0, hg - ag))
        self.form[a].append((3 if ag > hg else 1 if hg == ag else 0, ag - hg))
        self.count[h] += 1
        self.count[a] += 1


def training_rows(matches):
    state, rows, labels, dates = State(), [], [], []
    # Batch identical kickoffs: results cannot leak between simultaneous games.
    for date in sorted({m['utcDate'] for m in matches}):
        group = [m for m in matches if m['utcDate'] == date]
        for m in group:
            score = score90(m)
            if score is None:
                continue
            rows.append(state.features(m))
            labels.append('H' if score[0] > score[1] else 'D' if score[0] == score[1] else 'A')
            dates.append(date)
        for m in group:
            state.update(m)
    return np.array(rows), np.array(labels), dates, state


def estimator():
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=0.5))


def run(matches, now, days):
    valid = {}
    for m in matches:
        if m.get('competition', {}).get('code') not in COMPETITIONS:
            continue
        if not all(m.get(k, {}).get('id') is not None for k in ('homeTeam', 'awayTeam')):
            continue
        datetime.fromisoformat(m['utcDate'].replace('Z', '+00:00'))
        valid[m['id']] = m
    completed = [m for m in valid.values() if m['status'] == 'FINISHED'
                 and datetime.fromisoformat(m['utcDate'].replace('Z', '+00:00')) < now]
    X, y, dates, state = training_rows(completed)
    if len(y) < 60 or set(y) != set(LABELS):
        raise ValueError('Need at least 60 finished matches and all three outcomes. Fetch more history.')
    boundary = dates[int(len(dates) * 0.8)]
    train = np.array([d < boundary for d in dates])
    test = ~train
    if len(set(y[train])) != 3:
        raise ValueError('Chronological training partition lacks all outcomes. Fetch more history.')
    check = estimator().fit(X[train], y[train])
    prob = check.predict_proba(X[test])
    majority = max(LABELS, key=lambda label: sum(y[train] == label))
    report = {'training_matches': int(train.sum()), 'test_matches': int(test.sum()),
              'test_start_utc': boundary,
              'accuracy': float(accuracy_score(y[test], check.predict(X[test]))),
              'majority_baseline_accuracy': float(np.mean(y[test] == majority)),
              'log_loss': float(log_loss(y[test], prob, labels=check.classes_)),
              'evaluation': 'Chronological last 20%; form updates after each completed kickoff. No tuning on holdout.'}
    model = estimator().fit(X, y)
    rows = []
    for m in sorted(valid.values(), key=lambda m: m['utcDate']):
        kickoff = datetime.fromisoformat(m['utcDate'].replace('Z', '+00:00'))
        if m['status'] not in ('SCHEDULED', 'TIMED') or not now <= kickoff <= now + timedelta(days=days):
            continue
        p = dict(zip(model.classes_, model.predict_proba([state.features(m)])[0]))
        h, a = state.keys(m)
        rows.append({'competition': h[0], 'kickoff_utc': m['utcDate'],
                     'home': m['homeTeam']['name'], 'away': m['awayTeam']['name'],
                     'home_win': p['H'], 'draw': p['D'], 'away_win': p['A'],
                     'prediction': max(p, key=p.get), 'home_history': state.count[h],
                     'away_history': state.count[a],
                     'limited_history': min(state.count[h], state.count[a]) < 5})
    columns = ['competition', 'kickoff_utc', 'home', 'away', 'home_win', 'draw', 'away_win',
               'prediction', 'home_history', 'away_history', 'limited_history']
    return pd.DataFrame(rows, columns=columns), report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--season', type=int, help='Season start year, e.g. 2026 for 2026/27')
    parser.add_argument('--history', type=int, default=1, help='Seasons including current (default 1; older seasons may require paid access)')
    parser.add_argument('--days', type=int, default=14)
    parser.add_argument('--refresh', action='store_true', help='Replace API cache with latest fixtures/results')
    parser.add_argument('--cache', type=Path, default=Path('data/football'))
    parser.add_argument('--output', type=Path, default=Path('outputs/predictions.csv'))
    parser.add_argument('--input-json', type=Path, help='Offline football-data v4 JSON with a matches list')
    args = parser.parse_args()
    if args.history < 1 or args.days < 1:
        parser.error('--history and --days must be positive')
    now = datetime.now(timezone.utc)
    season = args.season if args.season is not None else now.year - (now.month < 7)
    try:
        if args.input_json:
            matches = json.loads(args.input_json.read_text(encoding='utf-8'))['matches']
        else:
            matches = []
            for year in range(season - args.history + 1, season + 1):
                for c in COMPETITIONS:
                    print(f'Loading {c} season {year}...', flush=True)
                    matches.extend(fetch(c, year, os.environ.get('FOOTBALL_DATA_TOKEN'), args.cache, args.refresh)['matches'])
        output, report = run(matches, now, args.days)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        output.to_csv(args.output, index=False)
        args.output.with_suffix('.metrics.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
        print(json.dumps(report, indent=2))
        print(f'Saved {len(output)} upcoming fixtures to {args.output}')
        if output.empty:
            print('No scheduled matches in this window. Increase --days or use --refresh.')
        print('Probabilities are estimates, not guaranteed results. H=home, D=draw, A=away; 90 minutes only.')
    except (ValueError, KeyError, OSError, requests.RequestException) as exc:
        print(f'Error: {exc}', file=sys.stderr)
        sys.exit(1)
