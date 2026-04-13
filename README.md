# Fraud Detection API

A production-ready REST API that detects fraudulent credit card transactions in real time. 
The system uses a tuned LightGBM classifier trained on 284,807 transactions and returns 
not just a prediction, but an explanation of why a transaction was flagged — making it 
useful for real fraud investigation workflows, not just binary classification.

---

## The Problem

Credit card fraud detection is not a standard classification problem. The dataset has a 
0.17% fraud rate — meaning for every 10,000 transactions, only 17 are fraudulent. A model 
that predicts "not fraud" on everything achieves 99.83% accuracy while being completely 
useless. This project addresses that directly using SMOTE oversampling and evaluation 
metrics that actually matter for imbalanced data.

---

## Model Performance

Four approaches were trained and evaluated on the same test set. F1 score on the fraud 
class was used as the primary metric rather than accuracy, since accuracy is misleading 
on heavily imbalanced data.

| Model              | F1 Score | Precision | Recall | Approach                  |
|--------------------|----------|-----------|--------|---------------------------|
| LightGBM (tuned)   | 0.8293   | 0.79      | 0.87   | Supervised + SMOTE        |
| LightGBM (base)    | 0.8000   | 0.75      | 0.86   | Supervised + SMOTE        |
| XGBoost            | 0.7556   | 0.67      | 0.87   | Supervised + SMOTE        |
| Isolation Forest   | 0.2634   | 0.25      | 0.28   | Unsupervised              |

The Isolation Forest result is intentional and informative — it shows that fraud patterns 
in this dataset are not simple statistical outliers. They require labeled supervision to 
detect effectively, which is why the supervised approaches outperform it significantly.

The tuned LightGBM model was selected as the production model. Hyperparameter tuning 
improved F1 from 0.80 to 0.83, primarily by reducing false positives while maintaining 
recall above 0.87.

---

## What the API Returns

Most fraud detection systems return a binary flag. This API returns a full explanation 
alongside the prediction, which is closer to what a real fraud operations team would need.

Each prediction response includes the fraud probability as a continuous score, a risk 
tier of HIGH, MEDIUM, or LOW based on that probability, and the top 5 features that 
drove that specific prediction using SHAP values. This means every flagged transaction 
comes with a reason, not just a label.

Sample response for a confirmed fraud transaction:

```json
{
  "is_fraud": true,
  "fraud_probability": 0.9994,
  "risk_level": "HIGH",
  "top_contributing_features": {
    "V13": 4.2286,
    "V9": 1.7214,
    "Amount_scaled": -1.6948,
    "V3": 1.3536,
    "V6": 1.1054
  },
  "processing_time_ms": 42.51
}
```

Average inference time is under 50ms per transaction.

---

## Project Structure

```
fraud-detection-api/
├── dataset/            # creditcard.csv dataset is present here (not tracked in git , downloaded from Kaggle)
├── models/             # Trained model and scaler artifacts (not tracked in git)
├── notebooks/          # EDA, training, and benchmarking notebook
├── src/
│   ├── main.py         # FastAPI application and route definitions
│   └── predict.py      # Preprocessing and prediction logic
├── tests/              # Unit tests
├── Dockerfile
└── requirements.txt
```

---

## Running the API

### With Docker (recommended)

```bash
docker build -t fraud-detection-api .
docker run -p 8000:8000 fraud-detection-api
```

### Without Docker

```bash
pip install -r requirements.txt
cd src
uvicorn main:app --reload
```

Once running, the interactive API documentation is available at:
http://127.0.0.1:8000/docs

---

## Dataset

The dataset used is the Credit Card Fraud Detection dataset published by the Machine 
Learning Group at Université Libre de Bruxelles, available on Kaggle at:
https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

Download creditcard.csv and place it in the /dataset directory. The file is not 
included in this repository due to its size (144MB).

The dataset contains 284,807 transactions made by European cardholders over two days 
in September 2013. Features V1 through V28 are principal components obtained via PCA 
to protect cardholder privacy. Time and Amount are the only original features.

---

## Technical Decisions

SMOTE was applied only to the training set after the train-test split to prevent data 
leakage. Applying it before splitting would cause synthetic samples derived from test 
data to appear in training, artificially inflating performance metrics.

Two separate scalers were fit for Amount and Time rather than a single scaler, since 
they have very different distributions and should be normalized independently.

SHAP TreeExplainer was used over model-agnostic explainers like LIME because it 
produces exact Shapley values for tree-based models rather than approximations, 
and is significantly faster at inference time.

---

## Tech Stack

- LightGBM, XGBoost, scikit-learn for modeling
- imbalanced-learn for SMOTE oversampling
- SHAP for prediction explainability
- FastAPI and Pydantic for the REST API
- Docker for containerization