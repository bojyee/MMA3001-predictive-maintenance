# Fixed-model performance profiling

**Captured (UTC):** 2026-10-24T17:34:03+00:00

**System:** Windows-11-10.0.26200-SP0  
**Python:** 3.13.3  
**scikit-learn:** 1.9.1  
**CPU count (logical):** 16

## Method

- All classifiers were kept at their **original Notebook 06 configurations**. No hyperparameters or thresholds were tuned.
- Each was fitted on the identical 8,000-row development subset. Prediction timing uses the same 2,000-row test features but does not use their labels.
- Each model was fitted **7** times, with **30** timed probability-prediction batches per fit. The table reports the median observed wall-clock time in milliseconds.
- `tracemalloc` was measured during a **separate fit** and counts Python-traced allocations only. **It is not total process RAM or full native-library peak memory.**

| Model | Median fit (ms) | Median prediction batch of 2,000 (ms) | Peak Python-traced allocations (MiB) |
|---|---:|---:|---:|
| Majority baseline | 0.592 | 0.158 | 0.317 |
| Logistic Regression (C=1) | 23.675 | 3.620 | 0.999 |
| Frozen Decision Tree | 26.593 | 3.681 | 1.109 |

## Engineering interpretation

These measurements compare **computational cost**, not prediction correctness. The majority baseline is computationally cheap but detects no failures. Evaluate performance and error consequences together with the notebook confusion matrices. Laptop/desktop hardware, thread settings and package versions affect the values; rerun locally on the actual submission computer for representative measurements.

## Reproduction

```powershell
python scripts/profile_models.py
```

**Important evaluation limit:** The previously inspected test subset is not an independent deployment evaluation. Running the fixed models for timing does not correct this limitation.
