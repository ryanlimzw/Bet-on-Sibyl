"""Proper scoring rules for three-way football outcome probabilities."""
import numpy as np


def multiclass_brier(y_true, probabilities, classes=('A', 'D', 'H')):
    p = np.asarray(probabilities, dtype=float)
    y = np.asarray(y_true)
    if p.shape != (len(y), len(classes)):
        raise ValueError('Invalid probability dimensions')
    if not np.all(np.isfinite(p)) or np.any(p < 0) or not np.allclose(p.sum(axis=1), 1):
        raise ValueError('Invalid probability values')
    actual = (y[:, None] == np.asarray(classes)[None, :]).astype(float)
    return float(np.mean(np.sum((p - actual) ** 2, axis=1)))
