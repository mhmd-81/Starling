import numpy as np

from starling.data_load import load_channel
from starling.detectors import trailing_deviation
from starling.evaluation import count_segments, event_metrics, point_metrics
from starling.splitter import time_split
from starling.thresholds import fit_threshold, flag


def run_channel(chan_id, window=101, q=0.99, fit_frac=0.8, data_dir="data"):
    """Run the static trailing-median baseline on one labeled channel."""
    train, test, intervals = load_channel(chan_id, data_dir=data_dir)

    # 1. threshold from the fit part of train only
    dev_fit, _ = time_split(trailing_deviation(train, window), fit_frac)
    if np.isnan(dev_fit).all():
        raise ValueError(f"{chan_id}: no valid fit scores (train too short for window={window})")
    threshold = fit_threshold(dev_fit, q)

    # 2. apply it, frozen, to test
    dev_test = trailing_deviation(test, window)
    pred = flag(dev_test, threshold)
    valid = ~np.isnan(dev_test)

    # 3. labels are used only here, for scoring
    truth = np.zeros(len(test), dtype=bool)
    for s, e in intervals:
        truth[s:e + 1] = True

    return {
        "params": {"chan_id": chan_id, "window": window, "q": q, "fit_frac": fit_frac},
        "threshold": threshold,
        "point": point_metrics(pred, truth, valid),
        "events": event_metrics(pred, intervals),
        "false_alarm_segments": count_segments(pred & ~truth),
    }