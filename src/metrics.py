import numpy as np


def per_vessel_errors(actual, pred):
    """Euclidean error in metres, shape (N, T)."""
    return np.linalg.norm(pred - actual, axis=-1)


def summarize(err, threshold_m):
    """ADE/FDE as in training. Accuracy = share of predicted waypoints within threshold_m of truth."""
    return {
        "ade": float(err.mean()),
        "fde": float(err[:, -1].mean()),
        "accuracy": float((err <= threshold_m).mean() * 100.0),
        "worst_fde": float(err[:, -1].max()),
        "vessels": int(err.shape[0]),
    }
