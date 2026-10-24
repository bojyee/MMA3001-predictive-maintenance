"""Integration check: the source modules reproduce saved Notebook 06 results."""

from pathlib import Path

import pytest

from src.data import load_dataset, make_original_split
from src.evaluation import evaluate_probabilities
from src.model import FINAL_THRESHOLD, create_final_model


def test_final_tree_matches_saved_test_confusion_matrix():
    dataset_path = Path("data/raw/ai4i2020.csv")
    if not dataset_path.is_file():
        pytest.skip("Download original AI4I CSV into data/raw to run the integration test")
    df = load_dataset(dataset_path)
    assert len(df) == 10_000 and int(df["Machine failure"].sum()) == 339
    X_train, X_test, y_train, y_test = make_original_split(df)
    assert (len(X_train), int(y_train.sum())) == (8000, 271)
    assert (len(X_test), int(y_test.sum())) == (2000, 68)
    model = create_final_model().fit(X_train, y_train)
    probability = model.predict_proba(X_test)[:, 1]
    metrics = evaluate_probabilities(y_test, probability, FINAL_THRESHOLD)
    assert (metrics["True Negatives"], metrics["False Positives"]) == (1921, 11)
    assert (metrics["False Negatives"], metrics["True Positives"]) == (24, 44)
    assert metrics["F1-score"] == pytest.approx(0.7154471545, abs=1e-6)
    assert metrics["PR-AUC"] == pytest.approx(0.7601, abs=1e-4)


def test_logistic_comparator_reproduces_notebook06():
    """The selected Logistic Regression settings and threshold remain untouched."""
    from scripts.profile_models import make_logistic

    path = Path("data/raw/ai4i2020.csv")
    if not path.is_file():
        pytest.skip("Original AI4I CSV required for integration comparison")
    df = load_dataset(path)
    X_train, X_test, y_train, y_test = make_original_split(df)
    logistic = make_logistic().fit(X_train, y_train)
    p = logistic.predict_proba(X_test)[:, 1]
    metrics = evaluate_probabilities(y_test, p, 0.20)
    assert (metrics["True Positives"], metrics["False Negatives"], metrics["False Positives"]) == (32, 36, 40)
    assert metrics["ROC-AUC"] == pytest.approx(0.8995, abs=1e-4)
    assert metrics["PR-AUC"] == pytest.approx(0.4215, abs=1e-4)


def test_frozen_prediction_interface_matches_notebook06():
    """The documented prediction helper uses the original probability threshold."""
    from src.model import predict_failure

    path = Path("data/raw/ai4i2020.csv")
    if not path.is_file():
        pytest.skip("Original AI4I CSV required for integration comparison")
    X_train, X_test, y_train, _ = make_original_split(load_dataset(path))
    model = create_final_model().fit(X_train, y_train)
    labels, probabilities = predict_failure(model, X_test)
    assert len(labels) == 2000
    assert int(labels.sum()) == 55  # 44 TP + 11 FP
    assert len(probabilities) == 2000
