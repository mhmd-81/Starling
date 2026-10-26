import numpy as np

from starling.data_load import load_channel
from starling.detectors import trailing_deviation
from starling.evaluation import auc_score, count_segments, event_metrics, point_metrics
from starling.models import Linear_model
from starling.splitter import time_split
from starling.thresholds import fit_threshold, flag


def build_scorer(method, fit_signal, window=101, lag=32):
    if method == "rolling_median":
        return lambda x: trailing_deviation(x, window)
    if method == "linear":
        return Linear_model(lag).fit(fit_signal).score
    raise ValueError(f"unknown method: {method}")


def run_channel(chan_id, method="rolling_median", window=101, lag=32,
                q=0.99, fit_frac=0.8, data_dir="data"):
    train, test, intervals = load_channel(chan_id, data_dir=data_dir)

    fit_signal, _ = time_split(train, fit_frac)
    scorer = build_scorer(method, fit_signal, window, lag)
    err_fit, _ = time_split(scorer(train), fit_frac)
    if np.isnan(err_fit).all():
        raise ValueError(f"{chan_id}: no valid fit scores (train too short)")
    threshold = fit_threshold(err_fit, q)

    test_scores = scorer(test)
    pred = flag(test_scores, threshold)
    valid = ~np.isnan(test_scores)

    truth = np.zeros(len(test), dtype=bool)
    for s, e in intervals:
        truth[s:e + 1] = True

    return {
        "params": {"chan_id": chan_id, "method": method, "window": window,
                   "lag": lag, "q": q, "fit_frac": fit_frac},
        "threshold": threshold,
        "point": point_metrics(pred, truth, valid),
        "auc": auc_score(test_scores, truth, valid),
        "events": event_metrics(pred, intervals),
        "false_alarm_segments": count_segments(pred & ~truth),
    }