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

from .features import EnhancedState, kickoff

COMPETITIONS = ('PL', 'CL')
DOMESTIC = ('BL1', 'PD', 'SA', 'FL1', 'DED', 'PPL')
DATA_COMPETITIONS = COMPETITIONS + DOMESTIC
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


def training_rows(matches, baseline=False):
    state, rows, labels, dates = State() if baseline else EnhancedState(), [], [], []
    if baseline:
        matches = [m for m in matches if m["competition"]["code"] in COMPETITIONS]
    # Use only prior calendar-day results, including across domestic/European games.
    groups = defaultdict(list)
    for m in matches:
        groups[kickoff(m).date()].append(m)
    for date in sorted(groups):
        group = sorted(groups[date], key=kickoff)
        for m in group:
            score = score90(m)
            if score is None or m["competition"]["code"] not in COMPETITIONS:
                continue
            rows.append(state.features(m))
            labels.append('H' if score[0] > score[1] else 'D' if score[0] == score[1] else 'A')
            dates.append(kickoff(m).isoformat())
        for m in group:
            state.update(m)
    return np.array(rows), np.array(labels), dates, state


def estimator():
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=0.5))


def fit_model(X, y, dates, weighted=False):
    model = estimator()
    if weighted:
        parsed = [datetime.fromisoformat(d) for d in dates]
        latest = max(parsed)
        weights = np.array([2 ** (-(latest - d).total_seconds() / 86400 / 365) for d in parsed])
        return model.fit(X, y, logisticregression__sample_weight=weights)
    return model.fit(X, y)


def evaluate(X, y, dates, completed, old_X):
    ordered = sorted([m for m in completed if score90(m) is not None
                      and m['competition']['code'] in COMPETITIONS], key=kickoff)
    # Stable input order for ties matches training_rows.
    codes = np.array([m['competition']['code'] for m in ordered])
    folds = []
    # Expanding-window tests, fixed before observing results; final 20% separate.
    for low, high in ((.5, .65), (.65, .8), (.8, 1.0)):
        start = dates[int(len(dates) * low)]
        end = dates[int(len(dates) * high)] if high < 1 else None
        train = np.array([d < start for d in dates])
        test = np.array([d >= start and (end is None or d < end) for d in dates])
        if not test.any() or len(set(y[train])) != 3:
            continue
        majority = max(LABELS, key=lambda label: sum(y[train] == label))
        fold = {'test_start_utc': start, 'test_end_exclusive_utc': end,
                'training_matches': int(train.sum()), 'test_matches': int(test.sum()), 'models': {}}
        for name, features, weighted in [('previous', old_X, False), ('enhanced', X, True)]:
            model = fit_model(features[train], y[train], np.array(dates)[train], weighted)
            probs = model.predict_proba(features[test])
            predictions = model.classes_[probs.argmax(axis=1)]
            results = {}
            for code in ('ALL', *COMPETITIONS):
                subset = np.ones(test.sum(), dtype=bool) if code == 'ALL' else codes[test] == code
                if not subset.any():
                    continue
                results[code] = {'matches': int(subset.sum()),
                                 'accuracy': float(accuracy_score(y[test][subset], predictions[subset])),
                                 'log_loss': float(log_loss(y[test][subset], probs[subset], labels=model.classes_)),
                                 'majority_baseline_accuracy': float(np.mean(y[test][subset] == majority))}
            fold['models'][name] = results
        folds.append(fold)
    if not folds:
        raise ValueError('Insufficient chronological history for evaluation.')
    return {'model': 'enhanced', 'total_target_matches': len(y),
            'data_competitions': sorted({m['competition']['code'] for m in completed}),
            'evaluation': 'Fixed expanding windows; last fold is final 20%. Earlier results update form; no tuning on these folds.',
            'form_half_life_days': 120, 'training_half_life_days': 365, 'folds': folds}


def run(matches, now, days, model_name='previous'):
    if model_name not in ('previous', 'enhanced'):
        raise ValueError('Unknown model specification.')
    valid = {}
    for m in matches:
        if m.get('competition', {}).get('code') not in DATA_COMPETITIONS:
            continue
        if not all(m.get(k, {}).get('id') is not None for k in ('homeTeam', 'awayTeam')):
            continue
        kickoff(m)
        valid[m['id']] = m
    completed = [m for m in valid.values() if m['status'] == 'FINISHED'
                 and kickoff(m) < now]
    X, y, dates, state = training_rows(completed)
    if len(y) < 60 or set(y) != set(LABELS):
        raise ValueError('Need at least 60 finished matches and all three outcomes. Fetch more history.')
    # Report both fixed model specifications on identical target fixtures.
    old_X, old_y, old_dates, old_state = training_rows(completed, baseline=True)
    if dates != old_dates or not np.array_equal(y, old_y):
        raise ValueError('Baseline and enhanced evaluation fixtures differ.')
    report = evaluate(X, y, dates, completed, old_X)
    report['model'] = model_name
    if model_name == 'previous':
        X, state = old_X, old_state
    model = fit_model(X, y, dates, weighted=model_name == 'enhanced')
    rows = []
    for m in sorted(valid.values(), key=lambda m: m['utcDate']):
        when = kickoff(m)
        if m['competition']['code'] not in COMPETITIONS or m['status'] not in ('SCHEDULED', 'TIMED') or not now <= when <= now + timedelta(days=days):
            continue
        p = dict(zip(model.classes_, model.predict_proba([state.features(m)])[0]))
        h, a = state.keys(m)
        rows.append({'competition': m['competition']['code'], 'kickoff_utc': m['utcDate'],
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
    parser.add_argument('--history', type=int, default=3, help='Seasons including current (default 3; older seasons may require paid access)')
    parser.add_argument('--days', type=int, default=14)
    parser.add_argument('--model', choices=('previous', 'enhanced'), default='previous',
                        help='Previous model remains default; enhanced features did not improve consistently')
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
                for c in DATA_COMPETITIONS:
                    print(f'Loading {c} season {year}...', flush=True)
                    matches.extend(fetch(c, year, os.environ.get('FOOTBALL_DATA_TOKEN'), args.cache, args.refresh)['matches'])
        output, report = run(matches, now, args.days, args.model)
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
