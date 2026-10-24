# Credit Card Fraud Detection

A machine learning project that estimates the probability of credit card fraud and flags potentially suspicious transactions for human review. It combines an imbalanced-classification workflow, threshold optimization, model evaluation, explainability, and a FastAPI web application.

> **Disclaimer:** This is an educational portfolio project, not a production banking system. Predictions are not proof of fraud, and the application does not automatically block transactions.

## Features

- Exploratory data analysis and preprocessing.
- Comparison of Logistic Regression and Decision Tree classifiers.
- Class-weighted training to address class imbalance.
- Threshold selection using validation data and illustrative review costs.
- Evaluation on a held-out test set.
- SHAP-based model explanations.
- FastAPI prediction API and browser-based interface.
- Fraud probability, predicted label, review flag, and decision threshold in the API response.

## Dataset

The project uses the Credit Card Fraud Detection dataset, which contains anonymized European card transactions collected over approximately two days.

| Property | Value |
|---|---:|
| Original transactions | 284,807 |
| Exact duplicate rows removed | 1,081 |
| Transactions after deduplication | 283,726 |
| Legitimate transactions | 283,253 |
| Fraudulent transactions | 473 |
| Fraud prevalence after deduplication | 0.1667% |

The features include `Time`, `Amount`, anonymized PCA-transformed variables (`V1`–`V28`), and the binary target `Class`.

The raw dataset is stored locally and should not be committed to the repository.

## Methodology

### 1. Exploratory data analysis

The analysis examines data quality, class imbalance, transaction amounts and time, duplicate rows, and feature relationships. Correlations are treated as descriptive evidence rather than proof of causation.

### 2. Preprocessing

- Remove exact duplicate rows while preserving the original data for exploration.
- Use stratified random training, validation, and test splits.
- Apply `RobustScaler` to `Time` and `Amount` only, fitting it on training data.
- Leave `V1`–`V28` unchanged.
- Save the fitted scaler and processed splits for reproducibility.

The split proportions are approximately 70% training, 15% validation, and 15% testing.

### 3. Model training

Two classifiers are compared:

- **Logistic Regression** with balanced class weights.
- **Decision Tree** with balanced class weights, maximum depth 5, and minimum leaf size 10.

No resampling is applied to the validation or test sets.

### 4. Threshold selection

The Logistic Regression decision threshold is selected using validation data and an illustrative cost model:

- False negative (missed fraud): **$1,000**
- False positive (legitimate transaction flagged for review): **$25**

These values are assumptions for comparing operating thresholds, not verified industry costs. The selected threshold is `0.982`. It is fixed before evaluating the test set and is not retuned using test results.

### 5. Explainability

SHAP is used to examine feature contributions to Logistic Regression predictions. The `V1`–`V28` variables are PCA-transformed features, so their contributions do not directly identify ordinary transaction attributes.

## Final test results

Logistic Regression evaluated at the selected threshold of `0.982`:

| Metric | Result |
|---|---:|
| Precision | 0.5135 |
| Recall | 0.8028 |
| F1-score | 0.6316 |
| Average precision | 0.6264 |
| True positives (TP) | 57 |
| False positives (FP) | 54 |
| False negatives (FN) | 14 |
| True negatives (TN) | 42,434 |

The model identified 57 of the 71 fraudulent transactions in the held-out test set, while missing 14. Of the 111 transactions flagged, 54 were legitimate.

Under the illustrative cost assumptions, the estimated test-set cost is:

\[
14 \times \$1{,}000 + 54 \times \$25 = \$15{,}350
\]

This is a hypothetical evaluation estimate, not a measured business saving or real-world financial impact.

## Project structure

```text
fraud-detection/
├── app/
│   ├── main.py
│   ├── templates/
│   │   └── index.html
│   └── static/
│       └── style.css
├── artifacts/
│   ├── logistic_regression.joblib
│   └── robust_scaler.joblib
├── data/
│   └── creditcard.csv          # Local only; not committed
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_models.ipynb
│   └── 04_evaluation.ipynb
├── reports/
│   └── ...                     # Evaluation reports and figures
├── tests/
├── .gitignore
├── requirements.txt
└── README.md
```

## Installation and setup

Python 3.11 is the environment used to run the application during development.

1. Clone or download this repository.
2. Place the dataset at `data/creditcard.csv` if you want to reproduce the notebooks.
3. Create and activate a virtual environment:

   **Windows PowerShell**
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

4. Install dependencies:

   ```powershell
   python -m pip install -r requirements.txt
   ```

5. Ensure the trained model and scaler files exist at:

   ```text
   artifacts/logistic_regression.joblib
   artifacts/robust_scaler.joblib
   ```

   The web application uses these saved artifacts and does not require the raw dataset to make predictions.

6. Start the application from the repository root:

   ```powershell
   python -m uvicorn app.main:app --reload
   ```

7. Open the local application in your browser:

   - Web interface: `http://127.0.0.1:8000/`
   - API documentation: `http://127.0.0.1:8000/docs`
   - Health check: `http://127.0.0.1:8000/health`

Use the versions and dependencies declared in `requirements.txt`. If a dependency is missing, install it in the active environment rather than relying on packages from another Python installation.

## API endpoints

| Method | Endpoint | Purpose |
|---|---|---|
| `GET` | `/` | Serves the web interface |
| `GET` | `/health` | Returns application health information and model threshold |
| `POST` | `/predict` | Returns a transaction prediction |
| `GET` | `/docs` | Interactive Swagger API documentation |

The `/predict` endpoint expects the 30 model features: `Time`, `V1`–`V28`, and `Amount`. It returns:

```json
{
  "fraud_probability": 0.151375,
  "predicted_label": "legitimate",
  "review_flag": false,
  "threshold": 0.982
}
```

This example illustrates the response format only. Actual output depends on the submitted transaction. The label and review flag are based on whether the estimated probability meets the configured threshold.

## Limitations and responsible use

- **Limited time coverage:** The dataset covers approximately 48 hours, so it cannot establish reliable performance across long-term changes in fraud behavior.
- **Random split:** Stratified random splitting can produce optimistic estimates when transactions are similar or temporally related. The results do not prove generalization to future transactions.
- **Anonymized features:** `V1`–`V28` are PCA-transformed variables with limited direct business interpretability.
- **Illustrative costs:** The false-negative and false-positive costs are assumptions and may not reflect actual operational expenses.
- **Imperfect predictions:** The model produces false positives and false negatives; a high probability is not proof of fraud.
- **No production controls:** The prototype does not provide a validated real-time banking integration, production monitoring, or an automated transaction-blocking system.

The project demonstrates a reproducible machine learning workflow and an API-based demo. Real-world deployment would require broader temporal evaluation, operational validation, security controls, monitoring, and human oversight.

## License and dataset attribution

Check the original dataset's terms and attribution requirements before redistributing it. This repository does not include the raw transaction dataset.