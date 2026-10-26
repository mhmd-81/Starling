import numpy as np
from sklearn.metrics import roc_auc_score

def point_metrics(pred, truth, valid=None):
    
    pred = np.asarray(pred, bool)
    truth = np.asarray(truth, bool)
    if valid is None:
        valid = np.ones(len(pred), bool)
    p, t = pred[valid], truth[valid]

    tp = int((p & t).sum())
    fp = int((p & ~t).sum())
    fn = int((~p & t).sum())
    tn = int((~p & ~t).sum())

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    fpr = fp / (fp + tn) if fp + tn else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": precision, "recall": recall, "f1": f1, "fpr": fpr}


def event_metrics(pred, intervals):

    pred = np.asarray(pred, bool)
    events = []
    for start, end in intervals:
        window = pred[start:end + 1]
        hits = np.where(window)[0]
        events.append({
            "interval": (start, end),
            "detected": bool(len(hits)),
            "delay": int(hits[0]) if len(hits) else None,
            "flagged_fraction": float(window.mean()),
        })
    return events


def count_segments(mask):
    mask = np.asarray(mask, bool)
    if mask.size == 0:
        return 0
    return int((mask[1:] & ~mask[:-1]).sum() + mask[0])


def auc_score(scores, truth, valid=None):

    scores = np.asarray(scores, float)
    truth = np.asarray(truth, bool)
    if valid is None:
        valid = ~np.isnan(scores)
    s, t = scores[valid], truth[valid]
    if t.all() or not t.any():
        return float("nan")
    return float(roc_auc_score(t, s))