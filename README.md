
# Machine Failure Classification — MMA3001

**MMA3001: Numerical Methods and Machine Learning**  
Individual Engineering Project — Benjamin Ooi

## 1. Project Overview

This project investigates machine failure classification using the UCI AI4I 2020 Predictive Maintenance Dataset.

The objective is to compare machine-learning approaches for identifying failure-associated operating conditions while minimising missed failures and unnecessary maintenance alarms.

This is a prototype classification study, not a validated real-time predictive maintenance system.

## 2. Dataset

**Source:** [UCI AI4I 2020 Predictive Maintenance Dataset](https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset)

The dataset contains 10,000 synthetic observations:

- Normal conditions: 9,661 (96.61%)
- Machine failures: 339 (3.39%)

**Target:** `Machine failure`

Six input features are used:

1. Type (L, M, H)
2. Air temperature [K]
3. Process temperature [K]
4. Rotational speed [rpm]
5. Torque [Nm]
6. Tool wear [min]

Identifiers and failure-mode indicators are excluded to reduce target-leakage risks.

The dataset is stored at `data/raw/ai4i2020.csv`.

## 3. Modelling Workflow

The investigation is documented in six independently executable Jupyter notebooks.

| Notebook | Description |
|---|---|
| 01_data_audit.ipynb | Dataset inspection, class imbalance and feature selection |
| 02_baseline_modelling.ipynb | Majority-class baseline and stratified split |
| 03_linear_regression.ipynb | Linear Regression benchmark |
| 04_logistic_regression.ipynb | Logistic Regression classification |
| 05_model_optimisation.ipynb | Decision Tree optimisation and sensitivity analysis |
| 06_final_model_validation.ipynb | Frozen-model evaluation and comparison |

The dataset is divided using an 80/20 stratified split with `random_state=42`.

Model selection uses development-data evidence, including cross-validation.

## 4. Selected Model

The selected classifier is a Decision Tree:

- `max_depth=9`
- `min_samples_leaf=10`
- `random_state=42`
- Classification threshold = 0.50

The model uses one-hot encoding for the categorical `Type` feature. Numerical inputs are passed through unchanged.

These settings are fixed in the final implementation.

## 5. Main Results

Results on the previously inspected 2,000-observation test subset:

| Metric | Logistic Regression | Decision Tree |
|---|---:|---:|
| Accuracy | 0.9620 | 0.9825 |
| Precision | 0.4444 | 0.8000 |
| Recall | 0.4706 | 0.6471 |
| F1-score | 0.4571 | 0.7154 |
| PR-AUC | 0.4215 | 0.7601 |
| ROC-AUC | 0.8995 | 0.8961 |
| False positives | 40 | 11 |
| False negatives | 36 | 24 |

The selected Decision Tree detects 12 more failures and generates 29 fewer false alarms than the Logistic Regression comparator.

Logistic Regression achieves a slightly higher ROC-AUC, so the Decision Tree does not outperform it on every metric.

## 6. Installation and Reproduction

Open Windows PowerShell from the repository root.

Create and activate a virtual environment if one does not already exist:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
python -m pip install -r requirements.txt
```

Place the original dataset at `data/raw/ai4i2020.csv`.

To open the modelling notebooks:

```powershell
python -m jupyter lab
```

To reproduce the frozen final model:

```powershell
python scripts/run_final_model.py
```

## 7. Automated Testing

Software verification uses `pytest`.

Run all tests and generate the test report:

```powershell
python scripts/generate_test_report.py
```

**Verification on 25 October 2026:**

- 22 automated tests passed.
- No failed tests.
- Final-model results reproduced the recorded Notebook 06 results.

Test reports:

- [Test Summary](docs/test_summary.txt)
- [JUnit Test Report](docs/test_report.xml)

Passing the tests verifies the checked implementation behaviour, not generalisation to real industrial machinery.

## 8. HTML Source-Code Documentation

HTML API documentation is generated using `pdoc`.

The documented source modules are:

- `src/data.py` — Dataset loading, validation and splitting
- `src/model.py` — Frozen Decision Tree construction and prediction
- `src/evaluation.py` — Classification metrics and evaluation

Regenerate the documentation using:

```powershell
python -m pdoc src.data src.model src.evaluation -o docs/pdoc
```

View the generated documentation:

[HTML API Documentation](docs/pdoc/index.html)

## 9. Computational Profiling

The project includes a profiling script to compare training runtime, prediction runtime and memory requirements for existing models.

Run:

```powershell
python scripts/profile_models.py
```

Measurements should be reproduced on the target computer before drawing final performance conclusions.

## 10. Limitations

- The AI4I dataset is synthetic and may not represent actual industrial conditions.
- The classifier identifies failure-associated observations but does not predict time until failure.
- The test subset was inspected during earlier development and is not completely independent of model selection history.
- The selected classifier missed 24 of 68 recorded failures in the test subset.
- Real maintenance costs, downtime and false-alarm consequences are unavailable.
- External validation on real machinery has not been performed.

These limitations prevent claims of production-ready reliability.

## 11. AI Assistance

AI tools supported aspects of coding, code review, troubleshooting, documentation and project planning.

Material AI contributions, verification procedures, limitations and individual learning will be disclosed in the written project report.

All submitted results and engineering conclusions remain the author's responsibility.

## 12. Dataset Attribution and Licence

AI4I 2020 Predictive Maintenance Dataset (2020). UCI Machine Learning Repository.

DOI: https://doi.org/10.24432/C5HS5C

The dataset is distributed under CC BY 4.0. The project's own source-code licence is specified separately in `LICENSE`.
