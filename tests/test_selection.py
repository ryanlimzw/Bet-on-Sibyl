import json
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from modern_football.external import EloHistory, attach_xg, understat_results
from modern_football.features import AdvancedState
from modern_football.predict import run, training_rows, probability_metrics
from modern_football.selection import boundaries, select, temperature
from test_modern_football import match


class SelectionTests(unittest.TestCase):
    def history(self, n=180):
        start = datetime(2024, 1, 1, tzinfo=timezone.utc)
        return [match(i, (start + timedelta(days=i)).isoformat(), home=i % 8,
                      away=(i+1) % 8, score=((1, 0), (0, 0), (0, 1))[i % 3],
                      competition=('PL', 'CL')[i % 2]) for i in range(n)]

    def arrays(self, matches):
        old, y, dates, _ = training_rows(matches, baseline=True)
        enhanced = training_rows(matches)[0]
        advanced = training_rows(matches, advanced=True)[0]
        return dict(previous=old, enhanced=enhanced, advanced=advanced), y, dates

    def test_tuning_and_calibration_do_not_use_outer_labels(self):
        arrays, y, dates = self.arrays(self.history())
        cut = 144
        prefix = {k: v[:cut] for k, v in arrays.items()}
        before = select(prefix, y[:cut], dates[:cut])
        y[cut:] = 'H'
        after = select(prefix, y[:cut], dates[:cut])
        self.assertEqual(before.details, after.details)
        test = {k: v[cut:] for k, v in arrays.items()}
        np.testing.assert_allclose(before.predict_proba(test), after.predict_proba(test))
        # Calibration labels may change temperature, never coefficients or tuning selection.
        cal = boundaries(dates[:cut], (.8,))[0]
        y[cal:cut] = 'D'
        recalibrated = select(prefix, y[:cut], dates[:cut])
        self.assertEqual(before.details['members'], recalibrated.details['members'])
        for (_, m1), (_, m2) in zip(before.members, recalibrated.members):
            np.testing.assert_allclose(m1[-1].coef_, m2[-1].coef_)

    def test_day_boundaries(self):
        dates = ['2025-01-01T12:00:00+00:00', '2025-01-01T17:00:00+00:00',
                 '2025-01-02T12:00:00+00:00', '2025-01-03T12:00:00+00:00']
        self.assertEqual(boundaries(dates, (.25, .6)), [0, 2])

    def test_advanced_same_day_and_future_isolation(self):
        games = self.history(3)
        games[1]['utcDate'] = games[0]['utcDate']
        games[1]['competition'] = games[0]['competition']
        X = training_rows(games, advanced=True)[0]
        np.testing.assert_equal(X[0], X[1])
        games[-1]['score']['fullTime'] = {'home': 12, 'away': 0}
        np.testing.assert_equal(X, training_rows(games, advanced=True)[0])
        state = AdvancedState()
        state.update(games[0])
        near = match(10, '2024-01-03T12:00:00Z', home=0, away=1)
        far = match(11, '2024-03-03T12:00:00Z', home=0, away=1)
        self.assertLess(state.features(near)[14], state.features(far)[14])
        self.assertGreater(state.features(near)[15], state.features(far)[15])

    def test_external_xg_only_after_available_day(self):
        game = match(1, '2025-01-01T12:00:00Z')
        with TemporaryDirectory() as folder:
            p = Path(folder)/'xg.csv'
            p.write_text('match_id,home_team_id,away_team_id,kickoff_utc,home_xg,away_xg,available_at_utc\n'
                         '1,1,2,2025-01-01T12:00:00Z,4.2,0.1,2025-01-04T00:00:00Z\n')
            games, count = attach_xg([game], p)
            self.assertEqual(count, 1)
            state = AdvancedState()
            state.update(games[0])
            early = match(2, '2025-01-04T12:00:00Z')
            late = match(3, '2025-01-05T12:00:00Z')
            self.assertEqual(state.features(early)[22], 0)
            self.assertGreater(state.features(late)[22], 0)
            p.write_text(p.read_text().replace('1,1,2,', '1,8,2,'))
            with self.assertRaisesRegex(ValueError, 'team IDs'):
                attach_xg([game], p)

    def test_external_elo_point_in_time_and_expiry(self):
        with TemporaryDirectory() as folder:
            p = Path(folder)/'elo.csv'
            p.write_text('team_id,effective_at_utc,available_at_utc,elo\n'
                         '1,2025-01-01T00:00:00Z,2025-01-03T00:00:00Z,1800\n'
                         '1,2025-02-01T00:00:00Z,2025-02-01T00:00:00Z,2000\n')
            elo = EloHistory(p)
            self.assertEqual(elo.get(1, datetime(2025,1,3,tzinfo=timezone.utc))[1], 0)
            self.assertEqual(elo.get(1, datetime(2025,1,4,tzinfo=timezone.utc))[0], 1800)
            self.assertEqual(elo.get(1, datetime(2025,3,5,tzinfo=timezone.utc))[1], 0)
            self.assertEqual(elo.get(99, datetime(2025,1,4,tzinfo=timezone.utc))[1], 0)

    def test_public_csv_bom_deduplication_and_result_validation(self):
        from modern_football.public_csv import convert
        with TemporaryDirectory() as folder:
            p = Path(folder)/'results.csv'
            data = ('Div,Date,Time,HomeTeam,AwayTeam,FTHG,FTAG,FTR,HS\n'
                    'E0,18/08/2024,16:30,A,B,1,0,H,99\n')
            p.write_text('\ufeff'+data, encoding='utf-8')
            rows = convert([p,p])
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]['utcDate'], '2024-08-18T15:30:00+00:00')
            self.assertNotIn('HS', rows[0])
            p.write_text(data.replace(',H,99', ',A,99'))
            with self.assertRaisesRegex(ValueError, 'disagreement'):
                convert([p])

    def test_understat_parser_rejects_maintenance(self):
        rows = [{'id': '1', 'h': {'title': 'Arsenal'}, 'xG': {'h': '1.2', 'a': '.8'}}]
        encoded = ''.join('\\x%02x' % b for b in json.dumps(rows).encode())
        self.assertEqual(understat_results("var datesData = JSON.parse('"+encoded+"');"), rows)
        with self.assertRaisesRegex(ValueError, 'payload missing'):
            understat_results('<html>We will come back soon</html>')

    def test_tuned_forecast_and_probability_metrics(self):
        history = self.history()
        now = datetime(2024, 7, 1, tzinfo=timezone.utc)
        future = match(999, (now+timedelta(days=1)).isoformat(), home=99, competition='CL')
        future['status'] = 'SCHEDULED'
        output, report = run(history+[future], now, 14, 'tuned')
        self.assertEqual(len(output), 1)
        self.assertTrue(output.iloc[0]['limited_history'])
        self.assertAlmostEqual(output.iloc[0][['home_win','draw','away_win']].sum(), 1)
        for fold in report['folds']:
            self.assertIn('tuned', fold['models'])
            self.assertLess(fold['selection']['calibration_start_utc'][:10], fold['test_start_utc'])
        p = temperature(np.array([[.2, .3, .5]]), 2)
        self.assertGreater(p[0, 0], .2)
        self.assertAlmostEqual(p.sum(), 1)
        result = probability_metrics(np.array(['H']), np.array([[0.,0.,1.]]), np.array(['PL']), 'H')['ALL']
        self.assertEqual(result['brier_score'], 0)
        self.assertEqual(result['expected_calibration_error'], 0)


if __name__ == '__main__':
    unittest.main()
