"""Benchmark *existing* Notebook 06 classifiers; no tuning or new models.

Two separate observations are collected:
  - Median fit/predict wall-clock time using perf_counter (repeats).
  - Peak Python-traced allocations using tracemalloc (separate fit).

tracemalloc misses some native NumPy/scikit-learn allocations; its peak is not
whole-process maximum RAM. Timings depend on the machine and software versions.
"""

import csv
from datetime import datetime, timezone
import os
from pathlib import Path
import platform
import statistics
import sys
from time import perf_counter
import tracemalloc

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import sklearn  # noqa: E402
from sklearn.compose import ColumnTransformer  # noqa: E402
from sklearn.dummy import DummyClassifier  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.preprocessing import OneHotEncoder, StandardScaler  # noqa: E402

from src.data import NUMERIC_FEATURES, CATEGORICAL_FEATURES, RANDOM_STATE, load_dataset, make_original_split  # noqa: E402
from src.model import create_final_model  # noqa: E402


def make_logistic():
    """Rebuild Notebook 06's selected Logistic Regression comparator unchanged."""
    return Pipeline([
        ("preprocessing", ColumnTransformer([
            ("numeric", StandardScaler(), NUMERIC_FEATURES),
            ("categorical", OneHotEncoder(drop="first", handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ])),
        ("model", LogisticRegression(C=1, max_iter=1000, random_state=RANDOM_STATE)),
    ])


def benchmark(name, factory, X_train, X_test, y_train, repetitions=7, prediction_repetitions=30):
    """Measure fixed-configuration fit time and batch prediction latency."""
    fit_times_ms = []
    pred_times_ms = []
    for _ in range(repetitions):
        model = factory()
        start = perf_counter()
        model.fit(X_train, y_train)
        fit_times_ms.append((perf_counter() - start) * 1000)
        # Predictions are warmed once; the measured operation is predict_proba on
        # all 2,000 test observations (not per single observation).
        model.predict_proba(X_test)
        for _ in range(prediction_repetitions):
            start = perf_counter()
            model.predict_proba(X_test)
            pred_times_ms.append((perf_counter() - start) * 1000)

    # Python tracing is performed outside timing measurements to avoid bias.
    tracemalloc.start()
    fresh = factory()
    fresh.fit(X_train, y_train)
    _, peak_traced = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return {
        "model": name,
        "median_fit_ms": round(statistics.median(fit_times_ms), 3),
        "median_batch_predict_ms": round(statistics.median(pred_times_ms), 3),
        "median_predict_ms_per_row": round(statistics.median(pred_times_ms) / len(X_test), 6),
        "peak_traced_python_MiB": round(peak_traced / (1024 ** 2), 3),
        "fit_repetitions": repetitions,
        "prediction_repetitions_per_fit": prediction_repetitions,
    }


def main():
    """Write CSV measurements and an engineering interpretation report."""
    data = load_dataset(ROOT / "data" / "raw" / "ai4i2020.csv")
    X_train, X_test, y_train, _ = make_original_split(data)
    results = [
        benchmark("Majority baseline", lambda: DummyClassifier(strategy="most_frequent", random_state=RANDOM_STATE), X_train, X_test, y_train),
        benchmark("Logistic Regression (C=1)", make_logistic, X_train, X_test, y_train),
        benchmark("Frozen Decision Tree", create_final_model, X_train, X_test, y_train),
    ]
    output_dir = ROOT / "docs"
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "profiling_results.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)
    markdown_rows = "\n".join(
        f'| {r["model"]} | {r["median_fit_ms"]:.3f} | {r["median_batch_predict_ms"]:.3f} | {r["peak_traced_python_MiB"]:.3f} |'
        for r in results
    )
    report = f"""# Fixed-model performance profiling

**Captured (UTC):** {datetime.now(timezone.utc).isoformat(timespec='seconds')}

**System:** {platform.platform()}  
**Python:** {sys.version.split()[0]}  
**scikit-learn:** {sklearn.__version__}  
**CPU count (logical):** {os.cpu_count()}

## Method

- All classifiers were kept at their **original Notebook 06 configurations**. No hyperparameters or thresholds were tuned.
- Each was fitted on the identical 8,000-row development subset. Prediction timing uses the same 2,000-row test features but does not use their labels.
- Each model was fitted **7** times, with **30** timed probability-prediction batches per fit. The table reports the median observed wall-clock time in milliseconds.
- `tracemalloc` was measured during a **separate fit** and counts Python-traced allocations only. **It is not total process RAM or full native-library peak memory.**

| Model | Median fit (ms) | Median prediction batch of 2,000 (ms) | Peak Python-traced allocations (MiB) |
|---|---:|---:|---:|
{markdown_rows}

## Engineering interpretation

These measurements compare **computational cost**, not prediction correctness. The majority baseline is computationally cheap but detects no failures. Evaluate performance and error consequences together with the notebook confusion matrices. Laptop/desktop hardware, thread settings and package versions affect the values; rerun locally on the actual submission computer for representative measurements.

## Reproduction

```powershell
python scripts/profile_models.py
```

**Important evaluation limit:** The previously inspected test subset is not an independent deployment evaluation. Running the fixed models for timing does not correct this limitation.
"""
    (output_dir / "profiling_report.md").write_text(report, encoding="utf-8")
    print(report)
    print(f"Saved: {output_dir / 'profiling_results.csv'}")


if __name__ == "__main__":
    main()
