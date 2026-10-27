# AI-Based Placement Prediction System

A beginner-friendly full-stack placement classifier built with Flask, pandas, NumPy, and scikit-learn. The Flask app serves a responsive dashboard and JSON APIs. The dashboard reads the configured CSV, builds inputs from the detected model feature schema, compares model evaluation results, and runs predictions with the selected saved pipeline.

## What the application does

1. Reads the configured CSV and normalizes whitespace around column names.
2. Finds a placement target from common names such as `Placement Tomorrow` or `Placement Outcome`. If more than one column could be the target, it stops with an error and asks you to configure `PLACEMENT_TARGET`.
3. Supports binary target values encoded as `Yes/No`, `True/False`, or `1/0`.
4. Drops exact duplicate rows and rows without a valid target. Missing feature values are imputed inside each scikit-learn pipeline. Date/time and identifier columns (including names such as `Student ID`) are shown in the dataset profile but excluded from model inputs.
5. When a usable date column exists, splits whole dates chronologically into 70% train, 15% validation, and 15% test partitions. The newest dates are reserved for testing; rows sharing a date never cross partitions. The date is used only to create the split and is excluded from model features. If there is no usable date column, the app falls back to a stratified random split.
6. Fits imputers, scaling, and categorical encoding inside each model pipeline using training rows only.
7. Trains Decision Tree, Random Forest, Logistic Regression, calibrated Linear SVM, K-Nearest Neighbors, and Gradient Boosting models. A model that fails is reported with its own error while other models can still be used.
8. Selects the dashboard's best model using validation F1 for the placement class. Accuracy, precision, recall, F1-score, ROC-AUC, and confusion matrices are then reported on the held-out test partition.
9. Saves trained pipelines in `models.joblib`. The cache is reused only when the dataset contents, target setting, and scikit-learn version still match. The first training run can take a few minutes.

## Dataset

The default CSV is synthetic demo data with placement-related features such as CGPA, aptitude, coding, communication, attendance, internships, and interview score. Its labels are sampled from a smooth probability curve around a score rule, so small input changes can affect estimated probabilities. These are not real student outcomes; test scores on this file do not demonstrate real-world placement accuracy. For meaningful predictions, configure `PLACEMENT_DATA_PATH` to a representative historical placement CSV with verified outcomes.

To use another CSV, set `PLACEMENT_DATA_PATH` before starting Flask. If the target name is not one of the recognized placement names, or if the CSV has multiple possible target columns, set `PLACEMENT_TARGET` explicitly. Legacy `RAINFALL_DATA_PATH` and `RAINFALL_TARGET` still work too. These PowerShell settings apply to the current terminal:

```powershell
$env:PLACEMENT_DATA_PATH = "C:\data\placement.csv"
$env:PLACEMENT_TARGET = "Placement Tomorrow"
```

Feature columns are inferred from the selected CSV. Numeric columns use median imputation and scaling where appropriate; categorical columns use most-frequent imputation and one-hot encoding. New categorical values and numeric values outside the training range are accepted but surfaced as reliability warnings in prediction results. Predictions also show the selected model's held-out test performance and flag probabilities close to its decision threshold. These safeguards cannot make an unrepresentative dataset reliable; use verified, representative historical outcomes before relying on predictions.

## Set up and run in VS Code

Open this folder in VS Code, then open **Terminal → New Terminal**. Run:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python app.py
```

If PowerShell blocks activation, run the environment's Python directly instead:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe app.py
```

Open <http://127.0.0.1:5000>. Flask serves the frontend and the API from the same address. Stop the server with **Ctrl+C** in the terminal. To change the port, set `$env:PORT = "5050"` before `python app.py`.

## Dashboard controls

- **Reload dataset & retrain** rereads only the server-configured CSV, checks it, and trains fresh pipelines. The browser cannot supply a file path.
- **Retrain models** reruns model training and refreshes evaluation results.
- Select a metric to compare models and choose any model to display its confusion matrix. The prediction form offers Logistic Regression and SVM because they provide gradual probabilities for numeric inputs; tree and neighbor probabilities change in steps.
- Prediction inputs and defaults are generated from the dataset's actual feature schema. Numeric fields show the observed range; out-of-range values are allowed but produce a reliability warning. New categories similarly produce a warning where the observed category list is available.

## API endpoints

| Method | Route | Purpose |
|---|---|---|
| `GET` | `/api/health` | Check service and dataset availability |
| `GET` | `/api/dataset` | Return columns, types, missing counts, class distribution, and five source rows |
| `GET` | `/api/metrics` | Train or load models and return validation/test metrics |
| `POST` | `/api/predict` | Predict a placement outcome from feature values and a selected model |
| `POST` | `/api/retrain` | Retrain models on the configured dataset |
| `POST` | `/api/dataset/reload` | Reload the configured CSV and retrain |

Prediction JSON uses the exact input feature names returned by `/api/dataset`, plus an optional `model` field. Example for a placement dataset:

```json
{
  "model": "Random Forest",
  "Branch": "CSE",
  "CGPA": 8.7,
  "Aptitude Score": 82,
  "Coding Score": 90,
  "Communication Skills": 88,
  "Attendance %": 96,
  "Backlogs": 0,
  "Internships": 2,
  "Interview Score": 81
}
```

## Test the project

Install requirements first, then run the automated Flask test-client suite:

```powershell
python -m pytest tests -q -p no:cacheprovider
```

The tests cover the served page, API routes, prediction validation, missing dataset errors, duplicate removal, missing values, binary target encodings, and ambiguous target detection. Model training uses the configured CSV when `/api/metrics` is first requested.

## Observed results on the included CSV

One run on the synthetic demo CSV using a stratified random split produced these test metrics (1,800 test rows):

| Model | Accuracy | Precision | Placement recall | F1-score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Decision Tree | 81.00% | 75.04% | 75.25% | 75.15% | 82.08% |
| Random Forest | 91.00% | 90.08% | 85.88% | 87.93% | 97.12% |
| Logistic Regression | 93.33% | 89.99% | 92.87% | 91.40% | 98.70% |
| SVM | 94.44% | 93.61% | 91.70% | 92.65% | 98.70% |
| K-Nearest Neighbors | 89.06% | 87.58% | 83.11% | 85.29% | 95.67% |
| Gradient Boosting | 91.56% | 91.34% | 86.03% | 88.61% | 97.28% |

These figures measure recovery of the synthetic scoring rule only. Validate any model trained on real institutional data against future, representative cohorts before using its predictions.
