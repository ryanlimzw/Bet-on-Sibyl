"""Nested chronological tuning, blending and held-out temperature calibration."""
from dataclasses import dataclass

import numpy as np
from sklearn.metrics import log_loss
from .predict import LABELS, estimator


@dataclass(frozen=True)
class Spec:
    features: str
    c: float = .5
    half_life: int = 0


def boundaries(dates, fractions):
    """Split at UTC day boundaries so a day's labels never cross a split."""
    days = np.array([d[:10] for d in dates])
    return [int(np.searchsorted(days, days[min(int(len(days) * f), len(days)-1)]))
            for f in fractions]


def fit(spec, arrays, y, dates):
    model = estimator()
    model.set_params(logisticregression__C=spec.c)
    kwargs = {}
    if spec.half_life:
        from datetime import datetime
        parsed = [datetime.fromisoformat(d) for d in dates]
        weights = [2 ** (-(parsed[-1] - d).total_seconds()/86400/spec.half_life) for d in parsed]
        kwargs['logisticregression__sample_weight'] = weights
    return model.fit(arrays[spec.features], y, **kwargs)


def probability(model, X):
    p = model.predict_proba(X)
    return p[:, [list(model.classes_).index(k) for k in LABELS]]


def temperature(p, value):
    z = np.log(np.clip(p, 1e-12, 1)) / value
    z -= z.max(axis=1, keepdims=True)
    p = np.exp(z)
    return p / p.sum(axis=1, keepdims=True)


class SelectedModel:
    classes_ = np.array(LABELS)

    def __init__(self, members, weights, temp, details):
        self.members, self.weights, self.temp, self.details = members, weights, temp, details

    def predict_proba(self, arrays):
        p = sum(w * probability(model, arrays[spec.features])
                for (spec, model), w in zip(self.members, self.weights))
        return temperature(p, self.temp)


def select(arrays, y, dates):
    """Outer-test labels are never supplied here. Calibration is a separate suffix."""
    tune, calibration = boundaries(dates, (.6, .8))
    if tune < 20 or calibration <= tune or calibration >= len(y):
        raise ValueError('Need more distinct match days for nested chronological selection.')
    if len(set(y[:tune])) != 3:
        raise ValueError('Nested training prefix needs all three outcomes. Fetch more history.')
    specs = [Spec('previous')] + [Spec(name, c, half) for name in ('enhanced', 'advanced')
             for c in (.1, .5, 2) for half in (180, 365)]
    early = {k: v[:tune] for k, v in arrays.items()}
    validation = {k: v[tune:calibration] for k, v in arrays.items()}
    predictions = [probability(fit(s, early, y[:tune], dates[:tune]), validation[s.features]) for s in specs]
    losses = [log_loss(y[tune:calibration], p, labels=LABELS) for p in predictions]
    best = int(np.argmin(losses))
    # Include an unweighted previous baseline in the blend, selected on tuning only.
    options = [(1., predictions[best])]
    if best:
        options += [(w, w*predictions[best] + (1-w)*predictions[0]) for w in (.25, .5, .75)]
    weight, _ = min(options, key=lambda item: log_loss(y[tune:calibration], item[1], labels=LABELS))
    selected = [specs[best]] if weight == 1 else [specs[best], specs[0]]
    weights = [1.] if weight == 1 else [weight, 1-weight]
    development = {k: v[:calibration] for k, v in arrays.items()}
    members = [(s, fit(s, development, y[:calibration], dates[:calibration])) for s in selected]
    p = sum(w * probability(model, arrays[s.features][calibration:])
            for (s, model), w in zip(members, weights))
    temps = (1., .75, 1.25, 1.5, 2.)
    temp = min(temps, key=lambda t: log_loss(y[calibration:], temperature(p, t), labels=LABELS))
    details = {'members': [dict(features=s.features, C=s.c, half_life_days=s.half_life, weight=w)
                           for s, w in zip(selected, weights)],
               'temperature': temp, 'tuning_start_utc': dates[tune],
               'calibration_start_utc': dates[calibration],
               'coefficient_training_matches': calibration, 'calibration_matches': len(y)-calibration,
               'candidate_count': len(specs),
               'note': 'Coefficients fit before calibration suffix; no refit on calibration labels.'}
    return SelectedModel(members, weights, temp, details)
