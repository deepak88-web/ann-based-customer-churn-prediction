# Telco Customer Churn Prediction

A small end-to-end machine learning project that estimates whether a telecom customer is likely to leave. The project includes exploratory analysis, an ANN training workflow, reusable preprocessing, and a Streamlit interface for predictions from raw customer details.

## Project contents

```text
.
├── app.py
├── src/
│   ├── __init__.py
│   └── predict.py
├── artifacts/
│   └── production_pipeline/
│       ├── churn_ann.keras
│       ├── preprocessor.joblib
│       ├── feature_names.joblib
│       └── metadata.json
├── Churn Analysis - EDA.ipynb
├── Churn Analysis-Model_Building.ipynb
├── Churn Analysis-Production Pipeline.ipynb
├── WA_Fn-UseC_-Telco-Customer-Churn.csv
├── tel_churn.csv
└── requirements.txt
```

The EDA notebook explores and cleans the source data. The model-building notebook records the initial ANN baseline and decision tree comparison. The production-pipeline notebook starts from raw records, fits preprocessing using the training split, compares models on validation data, selects an ANN decision threshold, and saves the production artifacts.

## Run locally

The pinned dependencies in `requirements.txt` were used with Python 3.13.1.

In PowerShell, from the project folder:

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Streamlit prints a local address in the terminal and opens the app in a browser. Enter the customer details and select **Predict churn** to see the estimated probability and classification.

If PowerShell blocks environment activation, run the project interpreter directly instead:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

## How the prediction works

The app sends the entered customer fields to `src/predict.py`. The prediction function loads the saved ANN, fitted preprocessor, input-column metadata, and decision threshold from `artifacts/production_pipeline/`. The preprocessor standardizes continuous features, passes through `SeniorCitizen`, and one-hot encodes categorical features. Unknown categories are ignored by the saved encoder.

The ANN returns a churn probability. The saved threshold is approximately **0.47**; probabilities at or above it are labeled **Churn**. A probability is a model estimate, not a certainty.

## Model results

For the current ANN run, the recorded holdout metrics are:

| Metric | Result |
|---|---:|
| Accuracy | 79.5% |
| ROC AUC | 0.836 |
| PR AUC | 0.622 |
| Churn precision | 0.65 |
| Churn recall | 0.51 |
| Churn F1 | 0.57 |

The decision tree had somewhat higher churn recall and F1 at the default 0.50 threshold in the validation comparison, while the ANN had higher validation accuracy and ranking metrics. The selected ANN threshold prioritizes overall accuracy, so some churners are missed. The holdout split matches one examined during initial baseline development; these scores should be treated as development results, not as an untouched independent estimate.

## Rebuild the model artifacts

Open `Churn Analysis-Production Pipeline.ipynb` in Jupyter or VS Code and run its cells from top to bottom. It reads `WA_Fn-UseC_-Telco-Customer-Churn.csv` and writes the trained model, preprocessor, feature names, and metadata to `artifacts/production_pipeline/`.

## Current scope

- Predictions use the categories and ranges represented in the source dataset.
- The app is a local demo; it does not store submitted customer records.
- The model and artifacts must be included when deploying the app, or the artifacts must be provided through the deployment's configured storage.
- The notebooks use additional visualization and Jupyter packages beyond the app's pinned runtime requirements.
