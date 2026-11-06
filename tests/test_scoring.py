import unittest

from modern_football.scoring import multiclass_brier


class ScoringTests(unittest.TestCase):
    def test_perfect_predictions(self):
        self.assertEqual(multiclass_brier(['H', 'D'], [[0, 0, 1], [0, 1, 0]]), 0.0)

    def test_uniform_predictions(self):
        self.assertAlmostEqual(multiclass_brier(['A'], [[1/3, 1/3, 1/3]]), 2/3)

    def test_invalid_probabilities(self):
        with self.assertRaises(ValueError):
            multiclass_brier(['H'], [[0.2, 0.2, 0.2]])


if __name__ == '__main__':
    unittest.main()
