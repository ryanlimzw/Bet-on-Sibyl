"""Convert local football-data.co.uk Premier League CSVs to offline input JSON.

IDs are isolated from football-data.org IDs. Only date, kickoff, teams and results
are read; shots, odds and other post-match columns never become features.
"""
import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo


def convert(paths):
    matches = {}
    for path in paths:
        with Path(path).open(encoding='utf-8-sig', newline='') as handle:
            reader = csv.DictReader(handle)
            required = {'Div', 'Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR'}
            if not required.issubset(reader.fieldnames or []):
                raise ValueError(f'{path}: required result columns missing.')
            for row in reader:
                if row['Div'] != 'E0' or row['FTR'] not in ('H', 'D', 'A'):
                    continue
                date = datetime.strptime(row['Date'], '%d/%m/%Y' if len(row['Date']) == 10 else '%d/%m/%y')
                hour, minute = map(int, (row.get('Time') or '12:00').split(':'))
                date = date.replace(hour=hour, minute=minute, tzinfo=ZoneInfo('Europe/London')).astimezone(timezone.utc)
                h, a = row['HomeTeam'], row['AwayTeam']
                hg, ag = int(row['FTHG']), int(row['FTAG'])
                result = 'H' if hg > ag else 'D' if hg == ag else 'A'
                if min(hg, ag) < 0 or result != row['FTR']:
                    raise ValueError(f'{path}: score/result disagreement.')
                key = f'fdcsv:{date.date()}:{h}:{a}'
                match = dict(id=key, utcDate=date.isoformat(), status='FINISHED', stage='REGULAR_SEASON',
                             competition={'code': 'PL'}, homeTeam={'id': 'fdcsv:'+h, 'name': h},
                             awayTeam={'id': 'fdcsv:'+a, 'name': a},
                             score={'duration': 'REGULAR', 'fullTime': {'home': hg, 'away': ag}})
                if key in matches and matches[key] != match:
                    raise ValueError(f'{path}: conflicting duplicate fixture {key}')
                matches[key] = match
    return sorted(matches.values(), key=lambda m: m['utcDate'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('csv', nargs='+', type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    try:
        matches = convert(args.csv)
        if not matches:
            raise ValueError('No completed Premier League results found.')
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps({'matches': matches}), encoding='utf-8')
    except (ValueError, KeyError, OSError) as exc:
        parser.exit(1, f'Error: {exc}\n')
    print(f'Saved {len(matches)} completed Premier League results to {args.output}')


if __name__ == '__main__':
    main()
