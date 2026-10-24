"""Frozen model construction and inference-contract tests."""

import numpy as np
import pandas as pd
import pytest
from sklearn.exceptions import NotFittedError
from sklearn.tree import DecisionTreeClassifier

from src.model import FINAL_THRESHOLD, create_final_model, predict_failure
from tests.test_data import example_features


def test_frozen_pipeline_configuration():
    model = create_final_model()
    tree = model.named_steps["model"]
    assert isinstance(tree, DecisionTreeClassifier)
    assert tree.max_depth == 9
    assert tree.min_samples_leaf == 10
    assert tree.random_state == 42
    assert tree.class_weight is None
    assert FINAL_THRESHOLD == 0.50
    preprocessing = model.named_steps["preprocessing"]
    assert preprocessing.transformers[0][1] == "passthrough"
    encoder = preprocessing.transformers[1][1]
    assert encoder.drop == "first"
    assert encoder.handle_unknown == "ignore"


def test_prediction_matches_probability_cutoff():
    examples = pd.concat([example_features()] * 20, ignore_index=True)
    examples["Type"] = ["L", "M", "H", "L", "M"] * 4
    examples["Torque [Nm]"] = np.linspace(20, 80, 20)
    targets = np.array([0] * 10 + [1] * 10)
    model = create_final_model().fit(examples, targets)
    predictions, probabilities = predict_failure(model, examples)
    np.testing.assert_array_equal(predictions, (probabilities >= 0.5).astype(int))
    assert np.logical_and(0 <= probabilities, probabilities <= 1).all()


def test_unfitted_model_rejected():
    with pytest.raises(NotFittedError):
        predict_failure(create_final_model(), example_features())


def test_invalid_prediction_threshold_rejected():
    with pytest.raises(ValueError, match="threshold"):
        predict_failure(create_final_model(), example_features(), -0.01)
