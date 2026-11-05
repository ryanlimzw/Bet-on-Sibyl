"""Pre-match features built only from earlier results, using stable provider team IDs."""
from collections import defaultdict, deque
from datetime import datetime, timezone

import numpy as np


def kickoff(m):
    date = datetime.fromisoformat(m['utcDate'].replace('Z', '+00:00'))
    if date.tzinfo is None:
        raise ValueError('Match kickoff must include a timezone.')
    return date.astimezone(timezone.utc)


class EnhancedState:
    def __init__(self):
        self.ratings = defaultdict(lambda: 1500.0)
        self.count = defaultdict(int)
        self.form = defaultdict(lambda: deque(maxlen=30))
        self.venues = defaultdict(lambda: deque(maxlen=20))

    def keys(self, m):
        # One rating per team connects domestic leagues through European matches.
        return m['homeTeam']['id'], m['awayTeam']['id']

    def summary(self, records, date):
        # Prior equivalent to five matches prevents extreme ratings for new clubs.
        total = np.array([5 * 1.35, 5 * 1.35, 5 * 1.3], dtype=float)
        weight = 5.0
        for played, scored, conceded, points in records:
            w = 2 ** (-max(0, (date - played).total_seconds() / 86400) / 120)
            total += w * np.array([scored, conceded, points])
            weight += w
        return total / weight

    def features(self, m):
        h, a = self.keys(m)
        date = kickoff(m)
        hf, af = self.summary(self.form[h], date), self.summary(self.form[a], date)
        hv = self.summary(self.venues[h, 'H'], date)
        av = self.summary(self.venues[a, 'A'], date)
        return [(self.ratings[h] - self.ratings[a]) / 400,
                *hf, *af, *(hv - av),
                min(self.count[h], 30) / 30, min(self.count[a], 30) / 30,
                float(m['competition']['code'] == 'CL'), float(m.get('stage') == 'FINAL')]

    def update(self, m):
        from .predict import score90
        score = score90(m)
        if score is None:
            return
        hg, ag = score
        h, a = self.keys(m)
        advantage = 0 if m.get('stage') == 'FINAL' else 60
        expected = 1 / (1 + 10 ** ((self.ratings[a] - self.ratings[h] - advantage) / 400))
        actual = 1 if hg > ag else .5 if hg == ag else 0
        delta = 20 * (actual - expected)
        self.ratings[h] += delta
        self.ratings[a] -= delta
        date = kickoff(m)
        hr = (date, hg, ag, 3 if hg > ag else 1 if hg == ag else 0)
        ar = (date, ag, hg, 3 if ag > hg else 1 if hg == ag else 0)
        self.form[h].append(hr)
        self.form[a].append(ar)
        if m.get('stage') != 'FINAL':
            self.venues[h, 'H'].append(hr)
            self.venues[a, 'A'].append(ar)
        self.count[h] += 1
        self.count[a] += 1
