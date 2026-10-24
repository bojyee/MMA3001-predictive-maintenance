"""Reproduce Notebook 06 frozen Decision Tree results; do not tune anything."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data import load_dataset, make_original_split  # noqa: E402
from src.evaluation import evaluate_probabilities  # noqa: E402
from src.model import FINAL_THRESHOLD, create_final_model  # noqa: E402


def main() -> None:
    dataset = load_dataset(ROOT / "data" / "raw" / "ai4i2020.csv")
    X_train, X_test, y_train, y_test = make_original_split(dataset)
    model = create_final_model().fit(X_train, y_train)
    probabilities = model.predict_proba(X_test)[:, 1]
    metrics = evaluate_probabilities(y_test, probabilities, FINAL_THRESHOLD)
    print("Frozen Notebook 06 Decision Tree — previously inspected test subset")
    for name, value in metrics.items():
        print(f"{name}: {value:.6f}" if isinstance(value, float) else f"{name}: {value}")
    print("IMPORTANT: test results were inspected during historical model development.")


if __name__ == "__main__":
    main()
