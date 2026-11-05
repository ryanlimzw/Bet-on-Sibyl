import unittest
from datetime import datetime, timedelta, timezone
from unittest.mock import patch, Mock
from tempfile import TemporaryDirectory
from pathlib import Path
import numpy as np
from modern_football.predict import State, fetch, run, score90, training_rows
from modern_football.features import EnhancedState


def match(i, date, home=1, away=2, score=(1, 0), competition='PL'):
    return dict(id=i, utcDate=date, status='FINISHED', stage='REGULAR_SEASON',
                competition={'code': competition}, homeTeam={'id': home, 'name': f'Team {home}'},
                awayTeam={'id': away, 'name': f'Team {away}'},
                score={'duration': 'REGULAR', 'fullTime': dict(zip(('home', 'away'), score))})


class Tests(unittest.TestCase):
    def test_extra_time(self):
        m = match(1, '2025-01-01T12:00:00Z')
        m['score'] = {'duration': 'PENALTY_SHOOTOUT', 'fullTime': {'home': 5, 'away': 4}}
        self.assertIsNone(score90(m))
        m['score']['regularTime'] = {'home': 1, 'away': 1}
        self.assertEqual(score90(m), (1, 1))

    def test_no_result_leakage(self):
        matches = [match(1, '2025-01-01T12:00:00Z'), match(2, '2025-01-01T12:00:00Z'),
                   match(3, '2025-01-02T12:00:00Z')]
        X, _, _, _ = training_rows(matches)
        np.testing.assert_equal(X[0], X[1])
        matches[2]['score']['fullTime'] = {'home': 0, 'away': 8}
        np.testing.assert_equal(X, training_rows(matches)[0])

    def test_full_pipeline(self):
        start = datetime(2024, 1, 1, tzinfo=timezone.utc)
        history = [match(i, (start + timedelta(days=i)).isoformat(), home=i % 8,
                         away=(i+1) % 8, score=((1, 0), (0, 0), (0, 1))[i % 3],
                         competition=('PL', 'CL')[i % 2]) for i in range(150)]
        now = start + timedelta(days=151)
        fixture = match(999, (now + timedelta(days=1)).isoformat(), home=99, competition='CL')
        fixture['status'] = 'TIMED'
        history.append(fixture)
        df, metrics = run(history, now, 14, model_name='enhanced')
        previous, previous_metrics = run(history, now, 14)
        self.assertEqual(previous_metrics['model'], 'previous')
        self.assertEqual(len(previous), len(df))
        self.assertEqual(len(df), 1)
        self.assertAlmostEqual(df.iloc[0][['home_win', 'draw', 'away_win']].sum(), 1)
        self.assertTrue(df.iloc[0]['limited_history'])
        self.assertEqual(metrics['folds'][-1]['test_matches'], 30)
        history[-1]['status'] = 'CANCELLED'
        self.assertTrue(run(history, now, 14)[0].empty)

    def test_domestic_history_informs_cl_without_becoming_target(self):
        domestic = match(1, '2025-01-01T12:00:00Z', competition='PD', score=(4, 0))
        european = match(2, '2025-01-03T12:00:00Z', competition='CL')
        X, y, _, state = training_rows([domestic, european])
        self.assertEqual(len(y), 1)
        self.assertGreater(X[0, 0], 0)
        self.assertEqual(state.count[1], 2)
        # Future domestic results must not change earlier CL features.
        domestic['utcDate'] = '2025-01-04T12:00:00Z'
        np.testing.assert_equal(training_rows([domestic, european])[0], training_rows([european])[0])

    def test_overlapping_games_and_timezone_order(self):
        first = match(1, '2025-01-01T12:00:00Z', score=(5, 0))
        overlapping = match(2, '2025-01-01T13:00:00+00:00')
        later = match(3, '2025-01-02T12:00:00Z')
        X, _, _, _ = training_rows([later, overlapping, first])
        np.testing.assert_equal(X[0], X[1])
        self.assertGreater(X[2, 0], X[0, 0])

    def test_decay_venue_and_smoothing(self):
        state = EnhancedState()
        game = match(1, '2025-01-01T12:00:00Z', score=(5, 0))
        state.update(game)
        near = match(2, '2025-01-02T12:00:00Z')
        far = match(3, '2026-01-02T12:00:00Z')
        self.assertGreater(state.features(near)[1], state.features(far)[1])
        self.assertEqual(len(state.venues[1, 'H']), 1)
        self.assertEqual(len(state.venues[1, 'A']), 0)
        self.assertTrue(np.isfinite(state.features(far)).all())

    def test_insufficient_history(self):
        with self.assertRaisesRegex(ValueError, '60'):
            run([], datetime.now(timezone.utc), 14)

    @patch('modern_football.predict.requests.get')
    @patch('modern_football.predict.time.sleep')
    def test_api_and_cache(self, sleep, get):
        get.return_value = Mock(status_code=200)
        get.return_value.json.return_value = {'matches': []}
        with TemporaryDirectory() as directory:
            cache = Path(directory)
            fetch('PL', 2026, 'test-token', cache)
            fetch('PL', 2026, '', cache)
            self.assertEqual(get.call_count, 1)
            get.return_value.status_code = 403
            with self.assertRaisesRegex(ValueError, 'API plan'):
                fetch('CL', 2026, 'test-token', cache)


if __name__ == '__main__':
    unittest.main()
