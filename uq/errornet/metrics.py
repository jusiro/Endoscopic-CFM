import numpy as np
import sklearn.metrics as skm
from scipy import interpolate


def get_auroc(pos: np.ndarray, neg: np.ndarray) -> float:
    labels = [1] * len(pos) + [0] * len(neg)
    data = np.concatenate((pos, neg))
    auroc = skm.roc_auc_score(labels, data)
    return auroc


def get_fpr(pos: np.ndarray, neg: np.ndarray) -> float:
    labels = [1] * len(pos) + [0] * len(neg)
    data = np.concatenate((pos, neg))
    fpr, tpr, _ = skm.roc_curve(
        labels, data
    )  # data must be prob estimates / conf values of positive class
    return float(interpolate.interp1d(tpr, fpr)(0.95))