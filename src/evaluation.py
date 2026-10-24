"""Classification metrics with the exact conventions used in Notebook 06.

"PR-AUC" in this project uses scikit-learn average precision (AP), not
numerical trapezoidal integration of a precision-recall curve.
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score, average_precision_score, balanced_accuracy_score,
    confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score,
)


def evaluate_threshold(y_true, probabilities, threshold: float) -> dict:
    """Evaluate a class-1 probability score at a fixed decision threshold.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Actual labels 0 (normal) or 1 (failure).
    probabilities : array-like of shape (n_samples,)
        Failure-class probabilities in [0, 1].
    threshold : float
        A probability greater than or equal to this value predicts class 1.

    Returns
    -------
    dict
        Threshold, accuracy, balanced accuracy, precision, recall, F1,
        and TN/FP/FN/TP counts, matching Notebook 06's metric names.

    Raises
    ------
    ValueError
        On invalid labels, probabilities or threshold.

    Notes
    -----
    ``zero_division=0`` means no predicted failure produces precision 0,
    rather than a metric exception.
    """
    y, p = _validated_inputs(y_true, probabilities, threshold)
    prediction = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, prediction, labels=[0, 1]).ravel()
    return {
        "Threshold": float(threshold),
        "Accuracy": accuracy_score(y, prediction),
        "Balanced Accuracy": balanced_accuracy_score(y, prediction),
        "Precision": precision_score(y, prediction, zero_division=0),
        "Recall": recall_score(y, prediction, zero_division=0),
        "F1-score": f1_score(y, prediction, zero_division=0),
        "False Positives": int(fp),
        "False Negatives": int(fn),
        "True Positives": int(tp),
        "True Negatives": int(tn),
    }


def evaluate_probabilities(y_true, probabilities, threshold: float) -> dict:
    """Add ROC-AUC and average precision (PR-AUC) to threshold metrics.

    Parameters
    ----------
    y_true, probabilities, threshold
        As described in :func:`evaluate_threshold`.

    Returns
    -------
    dict
        Threshold metrics plus ``ROC-AUC`` and ``PR-AUC``.

    Raises
    ------
    ValueError
        If either class is absent (ranking metrics would be undefined), or
        if input validation fails.
    """
    y, p = _validated_inputs(y_true, probabilities, threshold)
    if len(np.unique(y)) != 2:
        raise ValueError("Both target classes are required for ROC-AUC and PR-AUC")
    metrics = evaluate_threshold(y, p, threshold)
    metrics["ROC-AUC"] = roc_auc_score(y, p)
    metrics["PR-AUC"] = average_precision_score(y, p)
    return metrics


def _validated_inputs(y_true, probabilities, threshold):
    """Return checked 1-D arrays; raise on inconsistent metric inputs."""
    if not np.isscalar(threshold) or not isinstance(threshold, (int, float, np.number)) or not np.isfinite(threshold):
        raise ValueError("threshold must be a finite scalar number")
    if not 0 <= threshold <= 1:
        raise ValueError("threshold must lie between 0 and 1")
    y = np.asarray(y_true)
    p = np.asarray(probabilities)
    if y.ndim != 1 or p.ndim != 1 or len(y) == 0 or y.shape != p.shape:
        raise ValueError("Labels and probabilities must be same-length nonempty 1-D arrays")
    try:
        valid_labels = np.isin(y, [0, 1]).all()
        p = p.astype(float)
    except (ValueError, TypeError):
        raise ValueError("Labels must be binary and probabilities numeric") from None
    if not valid_labels:
        raise ValueError("Labels must contain only 0 and 1")
    if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("Probabilities must be finite and between 0 and 1")
    return y.astype(int), p
