"""Unit tests for confusion-matrix orientation and threshold behaviour."""

import numpy as np
import pytest
from src.evaluation import evaluate_probabilities, evaluate_threshold


def test_confusion_matrix_counts_and_f1():
    # TN=2, FP=1, FN=1, TP=2.
    y = [0, 0, 0, 1, 1, 1]
    p = [0.1, 0.4, 0.8, 0.1, 0.6, 0.9]
    result = evaluate_threshold(y, p, 0.5)
    assert (result["True Negatives"], result["False Positives"]) == (2, 1)
    assert (result["False Negatives"], result["True Positives"]) == (1, 2)
    assert result["F1-score"] == pytest.approx(2 / 3)
    assert result["Recall"] == pytest.approx(2 / 3)
    assert result["Precision"] == pytest.approx(2 / 3)


def test_threshold_equality_predicts_failure():
    result = evaluate_threshold([0, 1], [0.1, 0.5], 0.5)
    assert result["True Positives"] == 1


def test_all_normal_predictions_zero_division():
    result = evaluate_threshold([0, 1], [0.0, 0.0], 0.5)
    assert result["Precision"] == 0
    assert result["Recall"] == 0
    assert result["False Negatives"] == 1


def test_ranking_scores_present_and_bounded():
    result = evaluate_probabilities([0, 1, 0, 1], [0.1, 0.9, 0.2, 0.7], 0.5)
    assert result["PR-AUC"] == pytest.approx(1.0)
    assert result["ROC-AUC"] == pytest.approx(1.0)


@pytest.mark.parametrize("p,threshold", [([0.2, 1.2], 0.5), ([0.2, np.nan], 0.5), ([0.2, 0.8], 1.2)])
def test_invalid_scores_or_threshold_rejected(p, threshold):
    with pytest.raises(ValueError):
        evaluate_threshold([0, 1], p, threshold)


def test_single_class_roc_undefined():
    with pytest.raises(ValueError, match="Both target classes"):
        evaluate_probabilities([0, 0], [0.1, 0.2], 0.5)
