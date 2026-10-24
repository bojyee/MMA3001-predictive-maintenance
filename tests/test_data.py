"""Schema and reproducibility tests for the existing AI4I modelling workflow."""

import pandas as pd
import pytest

from src.data import FEATURES, NUMERIC_FEATURES, load_dataset, make_original_split, validate_features


def example_features():
    """One representative synthetic operating input; not a training record."""
    return pd.DataFrame({
        "Type": ["L"],
        "Air temperature [K]": [300.0],
        "Process temperature [K]": [310.0],
        "Rotational speed [rpm]": [1500],
        "Torque [Nm]": [40.0],
        "Tool wear [min]": [100],
    })


def test_feature_schema_and_order():
    checked = validate_features(example_features())
    assert checked.columns.tolist() == FEATURES
    assert len(NUMERIC_FEATURES) == 5
    assert "TWF" not in checked.columns


def test_missing_column_rejected():
    with pytest.raises(ValueError, match="Missing required"):
        validate_features(example_features().drop(columns=["Torque [Nm]"]))


@pytest.mark.parametrize("bad_type", ["X", None])
def test_invalid_type_rejected(bad_type):
    frame = example_features()
    frame.loc[0, "Type"] = bad_type
    with pytest.raises(ValueError, match="Type|missing"):
        validate_features(frame)


def test_nan_and_nonnumeric_rejected():
    frame = example_features()
    frame.loc[0, "Torque [Nm]"] = float("nan")
    with pytest.raises(ValueError, match="missing"):
        validate_features(frame)
    frame = example_features()
    frame["Torque [Nm]"] = ["not a number"]
    with pytest.raises(ValueError, match="numeric"):
        validate_features(frame)


def test_missing_file_and_invalid_target(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_dataset(tmp_path / "absent.csv")
    dataset = example_features().copy()
    dataset["Machine failure"] = [2]
    csv = tmp_path / "wrong.csv"
    dataset.to_csv(csv, index=False)
    with pytest.raises(ValueError, match="binary"):
        load_dataset(csv)


def test_original_split_repeatability(tmp_path):
    # This test uses a tiny controlled dataset and does not depend on the UCI download.
    dataset = pd.concat([example_features()] * 20, ignore_index=True)
    dataset["Machine failure"] = [0] * 10 + [1] * 10
    split_a = make_original_split(dataset)
    split_b = make_original_split(dataset)
    assert [len(x) for x in split_a] == [16, 4, 16, 4]
    assert split_a[0].index.tolist() == split_b[0].index.tolist()
    assert split_a[1].index.tolist() == split_b[1].index.tolist()
    assert int(split_a[3].sum()) == 2
