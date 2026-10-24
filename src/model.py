"""Reproduce the frozen Decision Tree pipeline from Notebook 06.

No optimisation, tuning, new classifier, or future-failure horizon is introduced.
This pipeline classifies AI4I failure-labelled operating observations only.
"""

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier

from .data import CATEGORICAL_FEATURES, NUMERIC_FEATURES, RANDOM_STATE, validate_features

FINAL_TREE_DEPTH = 9
FINAL_TREE_MIN_LEAF = 10
FINAL_THRESHOLD = 0.50


def create_final_model() -> Pipeline:
    """Construct (but do not fit) the exact frozen Decision Tree pipeline.

    Returns
    -------
    sklearn.pipeline.Pipeline
        Numeric features passed through unchanged, Type one-hot encoded with
        ``drop='first'`` and ``handle_unknown='ignore'``, then a Decision Tree
        with ``max_depth=9``, ``min_samples_leaf=10``, ``random_state=42``.

    Notes
    -----
    Identical to the preprocessing and estimator choices in Notebook 06.
    Call ``fit(X_train, y_train)`` only using the original development split.
    """
    preprocessing = ColumnTransformer(
        transformers=[
            ("numeric", "passthrough", NUMERIC_FEATURES),
            (
                "categorical",
                OneHotEncoder(drop="first", handle_unknown="ignore"),
                CATEGORICAL_FEATURES,
            ),
        ]
    )
    return Pipeline(
        steps=[
            ("preprocessing", preprocessing),
            (
                "model",
                DecisionTreeClassifier(
                    max_depth=FINAL_TREE_DEPTH,
                    min_samples_leaf=FINAL_TREE_MIN_LEAF,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def predict_failure(model: Pipeline, features, threshold: float = FINAL_THRESHOLD):
    """Classify input operating observations using a fitted model.

    Parameters
    ----------
    model : sklearn.pipeline.Pipeline
        An already fitted classifier with ``predict_proba``.
    features : pandas.DataFrame
        Model inputs in the units shown by their six column headings.
    threshold : float, default 0.5
        Frozen probability cut-off for predicting failure class 1. Use 0.5
        for the reported model; changing it changes the operating decision.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray]
        Predicted class labels (0 or 1) and class-1 probability scores.

    Raises
    ------
    ValueError
        If the threshold or features fail validation.
    sklearn.exceptions.NotFittedError
        If the model has not been fitted.
    """
    if (not np.isscalar(threshold) or
            not isinstance(threshold, (int, float, np.number)) or
            not np.isfinite(threshold) or not 0 <= threshold <= 1):
        raise ValueError("threshold must be a finite number between 0 and 1")
    X = validate_features(features)
    probabilities = model.predict_proba(X)[:, list(model.classes_).index(1)]
    predictions = (probabilities >= threshold).astype(int)
    return predictions, probabilities
