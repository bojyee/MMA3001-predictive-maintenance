"""Data input contracts and the original stratified data split.

Only variables available as operating measurements are used. Failure-mode flags
and identifiers are never included in the six-feature model input.

The rows of the UCI AI4I CSV must be left in their original order because the
frozen seeded split is defined on this ordering.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

TARGET = "Machine failure"
CATEGORICAL_FEATURES = ["Type"]
NUMERIC_FEATURES = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]
FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES
RANDOM_STATE = 42


def load_dataset(csv_path: str | Path = "data/raw/ai4i2020.csv") -> pd.DataFrame:
    """Load the original AI4I CSV and check the required modelling columns.

    Parameters
    ----------
    csv_path : str or pathlib.Path
        Path relative to the working directory, or an absolute path.

    Returns
    -------
    pandas.DataFrame
        Raw dataframe in its original row order (no columns deleted).

    Raises
    ------
    FileNotFoundError
        If the CSV path does not exist.
    ValueError
        If a required column, value or binary target is invalid.

    Notes
    -----
    Calling this function does not fit preprocessing or select a model.
    The dataset is synthetic; observed feature ranges are not safe operating limits.
    """
    path = Path(csv_path)
    if not path.is_file():
        raise FileNotFoundError(f"Dataset not found: {path.resolve()}")
    data = pd.read_csv(path)
    missing = [column for column in FEATURES + [TARGET] if column not in data.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    validate_features(data[FEATURES])
    targets = data[TARGET]
    if targets.isna().any() or not targets.isin([0, 1]).all():
        raise ValueError("Machine failure target must contain only binary labels 0 and 1")
    return data


def validate_features(features: pd.DataFrame) -> pd.DataFrame:
    """Check feature names, category values, finiteness and missing values.

    Parameters
    ----------
    features : pandas.DataFrame
        Dataframe containing the six model inputs. Other columns may exist.
        Numerical units must be K, rpm, Nm and min as indicated by headings.

    Returns
    -------
    pandas.DataFrame
        A copy with exactly six columns, in the fixed feature order.

    Raises
    ------
    TypeError
        If input is not a pandas DataFrame.
    ValueError
        If columns, entries, category values or numerical types are invalid.

    Notes
    -----
    This checks schema, not sensor calibration or whether a combination of
    measurements is physically plausible. The model should not be treated as
    validated outside its training-data support.
    """
    if not isinstance(features, pd.DataFrame):
        raise TypeError("features must be a pandas DataFrame")
    missing = [column for column in FEATURES if column not in features.columns]
    if missing:
        raise ValueError(f"Missing required feature columns: {missing}")
    selected = features.loc[:, FEATURES].copy()
    if selected.isna().any().any():
        raise ValueError("The six feature columns must not contain missing values")
    if not selected["Type"].isin(["L", "M", "H"]).all():
        raise ValueError("Type must be one of 'L', 'M' or 'H'")
    for column in NUMERIC_FEATURES:
        if not pd.api.types.is_numeric_dtype(selected[column]):
            raise ValueError(f"{column} must contain numeric values in its stated units")
        if not np.isfinite(selected[column].to_numpy(dtype=float)).all():
            raise ValueError(f"{column} must contain only finite values")
    return selected


def make_original_split(data: pd.DataFrame):
    """Reproduce the exact 80/20 stratified split from the modelling notebooks.

    Parameters
    ----------
    data : pandas.DataFrame
        Dataset in its original row order and with the six inputs plus target.

    Returns
    -------
    tuple
        ``(X_train, X_test, y_train, y_test)`` with 80% development and
        20% held-out rows using ``random_state=42`` and ``stratify=y``.

    Notes
    -----
    The test subset was inspected earlier in the project. This function does
    not restore independent test status or claim deployment validation.
    """
    missing = [column for column in FEATURES + [TARGET] if column not in data.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    X = validate_features(data[FEATURES])
    y = data[TARGET].copy()
    if y.isna().any() or not y.isin([0, 1]).all():
        raise ValueError("Machine failure target must contain only 0 and 1")
    return train_test_split(X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y)
