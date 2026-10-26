import numpy as np


def fit_threshold(scores, q=0.99):
    if not 0 < q <= 1:
        raise ValueError("q must be in (0, 1]")
    return float(np.nanquantile(scores, q))


def flag(scores, threshold):
    """True where score > threshold. NaN scores (no verdict yet) are never flagged."""
    scores = np.asarray(scores)
    return ~np.isnan(scores) & (scores > threshold)