import unittest
import numpy as np
from modern_football.goal_model import GoalModel


class GoalModelTests(unittest.TestCase):
    def test_probabilities(self):
        rng = np.random.default_rng(42)
        features = rng.normal(size=(100, 4))
        home = rng.poisson(1.5, size=100)
        away = rng.poisson(1.1, size=100)
        probabilities = GoalModel().fit(features, home, away).predict_proba(features[:5])
        self.assertEqual(probabilities.shape, (5, 3))
        np.testing.assert_allclose(probabilities.sum(axis=1), 1.0, atol=1e-10)
        self.assertTrue(np.all(probabilities >= 0))

    def test_invalid_max_goals(self):
        with self.assertRaises(ValueError):
            GoalModel(max_goals=2)
