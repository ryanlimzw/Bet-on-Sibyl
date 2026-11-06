"""Independent Poisson goal model and scoreline probabilities."""
import numpy as np
from scipy.stats import poisson
from sklearn.linear_model import PoissonRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


class GoalModel:
    def __init__(self, max_goals=10):
        if max_goals < 3:
            raise ValueError('max_goals must be at least 3')
        self.max_goals = max_goals
        self.home = make_pipeline(StandardScaler(), PoissonRegressor(alpha=1.0, max_iter=1000))
        self.away = make_pipeline(StandardScaler(), PoissonRegressor(alpha=1.0, max_iter=1000))

    def fit(self, features, home_goals, away_goals):
        self.home.fit(features, home_goals)
        self.away.fit(features, away_goals)
        return self

    def predict_proba(self, features):
        home_rates = self.home.predict(features)
        away_rates = self.away.predict(features)
        result = []
        for h, a in zip(home_rates, away_rates):
            h_probs = poisson.pmf(np.arange(self.max_goals), h)
            a_probs = poisson.pmf(np.arange(self.max_goals), a)
            h_probs = np.append(h_probs, poisson.sf(self.max_goals - 1, h))
            a_probs = np.append(a_probs, poisson.sf(self.max_goals - 1, a))
            matrix = np.outer(h_probs, a_probs)
            result.append([np.triu(matrix, 1).sum(), np.trace(matrix), np.tril(matrix, -1).sum()])
        return np.asarray(result)
