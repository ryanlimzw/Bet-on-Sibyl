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


class AdvancedState(EnhancedState):
    """Rest/congestion, optional prior-match xG and historical external ratings."""
    def __init__(self, elo_history=None):
        super().__init__()
        from .external import EloHistory
        self.elo_history = elo_history or EloHistory()
        self.played = defaultdict(lambda: deque(maxlen=40))
        self.xg = defaultdict(lambda: deque(maxlen=30))

    def features(self, m):
        h, a = self.keys(m)
        date = kickoff(m)
        base = super().features(m)
        def schedule(team):
            past = [d for d in self.played[team] if d.date() < date.date()]
            rest = min((date - past[-1]).total_seconds()/86400, 30) if past else 7
            return [rest / 14, sum((date-d).total_seconds()/86400 <= 14 for d in past)/5,
                    float(bool(past))]
        def expected(team):
            records = [r for r in self.xg[team] if r[4].date() < date.date()]
            summary = self.summary([r[:4] for r in records], date)
            return [*summary[:2], min(len(records), 20)/20]
        he, ae = self.elo_history.get(h, date), self.elo_history.get(a, date)
        return [*base, *schedule(h), *schedule(a), *expected(h), *expected(a),
                (he[0]-ae[0])/400 if he[1] and ae[1] else 0.,
                he[1], ae[1], he[2], ae[2]]

    def update(self, m):
        from .predict import score90
        if score90(m) is None:
            return
        super().update(m)
        h, a = self.keys(m)
        date = kickoff(m)
        self.played[h].append(date)
        self.played[a].append(date)
        if '_xg' in m:
            hg, ag, available = m['_xg']
            self.xg[h].append((date, hg, ag, 0, available))
            self.xg[a].append((date, ag, hg, 0, available))
