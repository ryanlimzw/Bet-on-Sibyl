"""Explicit provider-ID joins and timestamp-safe optional external data."""
import csv
import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path

import requests


def timestamp(value):
    d = datetime.fromisoformat(value.replace('Z', '+00:00'))
    if d.tzinfo is None:
        raise ValueError('External timestamps must include a timezone.')
    return d.astimezone(timezone.utc)


def number(value, nonnegative=False):
    n = float(value)
    if not math.isfinite(n) or (nonnegative and n < 0):
        raise ValueError('External values must be finite and xG nonnegative.')
    return n


def attach_xg(matches, path):
    """No fuzzy names or positional joins. A malformed or duplicate ID is an error."""
    by_id = {str(m['id']): dict(m) for m in matches}
    seen, count = set(), 0
    with Path(path).open(encoding='utf-8-sig', newline='') as f:
        for row in csv.DictReader(f):
            key = row['match_id']
            if key in seen:
                raise ValueError(f'Duplicate xG match ID: {key}')
            seen.add(key)
            if key not in by_id:
                continue
            m = by_id[key]
            if (str(m['homeTeam']['id']), str(m['awayTeam']['id'])) != (row['home_team_id'], row['away_team_id']):
                raise ValueError(f'xG team IDs do not match fixture {key}')
            if timestamp(row['kickoff_utc']) != timestamp(m['utcDate']):
                raise ValueError(f'xG kickoff does not match fixture {key}')
            available = timestamp(row['available_at_utc'])
            if available <= timestamp(m['utcDate']):
                raise ValueError('Post-match xG must become available after kickoff.')
            m['_xg'] = (number(row['home_xg'], True), number(row['away_xg'], True), available)
            count += 1
    return list(by_id.values()), count


class EloHistory:
    def __init__(self, path=None):
        self.records = {}
        seen = set()
        if path:
            with Path(path).open(encoding='utf-8-sig', newline='') as f:
                for row in csv.DictReader(f):
                    team = int(row['team_id'])
                    effective = timestamp(row['effective_at_utc'])
                    available = timestamp(row['available_at_utc'])
                    key = team, effective, available
                    if key in seen:
                        raise ValueError('Duplicate ClubElo snapshot.')
                    seen.add(key)
                    self.records.setdefault(team, []).append((effective, available, number(row['elo'])))
            for rows in self.records.values():
                rows.sort()

    def get(self, team, date):
        # Prior UTC day is deliberately conservative for date-only historical feeds.
        eligible = [r for r in self.records.get(team, [])
                    if r[0].date() < date.date() and r[1].date() < date.date()]
        if not eligible:
            return 1500., 0., 0.
        effective, _, elo = max(eligible, key=lambda r: (r[0], r[1]))
        age = (date - effective).total_seconds()/86400
        if age > 30:
            return 1500., 0., min(age, 365)/365
        return elo, 1., age/365


def understat_results(html):
    """Parse the site's embedded JSON without evaluating JavaScript."""
    match = re.search(r"datesData\s*=\s*JSON\.parse\('((?:\\.|[^'\\])*)'\)", html)
    if not match:
        raise ValueError('Understat result payload missing. Feed may be unavailable or changed.')
    encoded = match.group(1)
    # Understat uses ASCII hex escapes for the embedded JSON.
    decoded = re.sub(r'\\x([0-9a-fA-F]{2})', lambda m: chr(int(m.group(1), 16)), encoded)
    rows = json.loads(decoded)
    if not isinstance(rows, list):
        raise ValueError('Understat results must be a list.')
    return rows


def download_understat(league, season, output):
    if league not in ('EPL', 'La_liga', 'Bundesliga', 'Serie_A', 'Ligue_1'):
        raise ValueError('Unsupported Understat league.')
    r = requests.get(f'https://understat.com/league/{league}/{int(season)}', timeout=30)
    r.raise_for_status()
    rows = understat_results(r.text)
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text(json.dumps({'source': r.url, 'downloaded_at_utc': datetime.now(timezone.utc).isoformat(),
                                      'matches': rows}, indent=2), encoding='utf-8')
    return len(rows)


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Download raw Understat league match data for explicit ID mapping.')
    parser.add_argument('--league', required=True, choices=('EPL', 'La_liga', 'Bundesliga', 'Serie_A', 'Ligue_1'))
    parser.add_argument('--season', required=True, type=int)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    try:
        count = download_understat(args.league, args.season, args.output)
    except (ValueError, OSError, requests.RequestException) as exc:
        parser.exit(1, f'Error: {exc}\n')
    print(f'Saved {count} raw records to {args.output}. Map IDs and availability timestamps before importing.')


if __name__ == '__main__':
    main()
