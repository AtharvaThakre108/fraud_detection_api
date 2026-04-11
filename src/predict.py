import pickle
import numpy as np
import shap

# Load model and scalers
with open('../models/fraud_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('../models/amount_scaler.pkl', 'rb') as f:
    amount_scaler = pickle.load(f)

with open('../models/time_scaler.pkl', 'rb') as f:
    time_scaler = pickle.load(f)

explainer = shap.TreeExplainer(model)


def preprocess(data: dict) -> np.ndarray:
    amount_scaled = amount_scaler.transform([[data['Amount']]])[0][0]
    time_scaled = time_scaler.transform([[data['Time']]])[0][0]

    # Order must match exactly: V1-V28, Amount_scaled, Time_scaled
    features = [data[f'V{i}'] for i in range(1, 29)] + \
               [amount_scaled, time_scaled]

    return np.array(features).reshape(1, -1)


def predict(data: dict) -> dict:
    features = preprocess(data)

    prediction = model.predict(features)[0]
    probability = model.predict_proba(features)[0][1]

    shap_values = explainer.shap_values(features)
    if isinstance(shap_values, list):
        shap_vals = shap_values[1][0]
    else:
        shap_vals = shap_values[0]

    feature_names = ['Time_scaled'] + \
                    [f'V{i}' for i in range(1, 29)] + \
                    ['Amount_scaled']

    shap_dict = dict(zip(feature_names, shap_vals))
    top_features = dict(sorted(shap_dict.items(),
                               key=lambda x: abs(x[1]),
                               reverse=True)[:5])

    return {
        "is_fraud": bool(prediction),
        "fraud_probability": round(float(probability), 4),
        "risk_level": "HIGH" if probability > 0.7 else "MEDIUM" if probability > 0.3 else "LOW",
        "top_contributing_features": {k: round(float(v), 4) for k, v in top_features.items()}
    }