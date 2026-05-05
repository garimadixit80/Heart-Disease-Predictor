import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

import shap
import matplotlib.pyplot as plt
import joblib

# -----------------------------
# Load Dataset
# -----------------------------
heart_data = pd.read_csv("heart.csv")

# -----------------------------
# Features & Target
# -----------------------------
X = heart_data.drop(columns='target')   # ✅ FIXED
Y = heart_data['target']

feature_names = X.columns

# -----------------------------
# Train-Test Split
# -----------------------------
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, stratify=Y, random_state=2
)

# -----------------------------
# Scaling (only for Logistic Regression)
# -----------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# -----------------------------
# Models
# -----------------------------
lr_model = LogisticRegression(max_iter=1000)

rf_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=5,
    min_samples_split=10,
    random_state=2
)

xgb_model = XGBClassifier(
    n_estimators=100,
    max_depth=3,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric='logloss'
)

models = {
    "Logistic Regression": lr_model,
    "Random Forest": rf_model,
    "XGBoost": xgb_model
}

# -----------------------------
# Train & Evaluate
# -----------------------------
for name, model in models.items():
    
    if name == "Logistic Regression":
        model.fit(X_train_scaled, Y_train)
        preds = model.predict(X_test_scaled)
    else:
        model.fit(X_train, Y_train)
        preds = model.predict(X_test)
    
    print(f"\n{name}")
    print("Accuracy:", round(accuracy_score(Y_test, preds), 4))
    print("Precision:", round(precision_score(Y_test, preds), 4))
    print("Recall:", round(recall_score(Y_test, preds), 4))
    print("F1 Score:", round(f1_score(Y_test, preds), 4))
    print("Confusion Matrix:\n", confusion_matrix(Y_test, preds))

# -----------------------------
# Cross Validation
# -----------------------------
print("\nCross Validation Scores:")
cv_scores = cross_val_score(rf_model, X, Y, cv=5)
print("Random Forest CV Accuracy:", cv_scores.mean())

# -----------------------------
# Feature Importance
# -----------------------------
print("\nLogistic Regression Feature Importance:")
for name, coef in zip(feature_names, lr_model.coef_[0]):
    print(f"{name}: {coef:.4f}")

print("\nRandom Forest Feature Importance:")
rf_importance = rf_model.feature_importances_
sorted_idx = np.argsort(rf_importance)[::-1]

for i in sorted_idx:
    print(f"{feature_names[i]}: {rf_importance[i]:.4f}")

print("\nXGBoost Feature Importance:")
xgb_importance = xgb_model.feature_importances_
sorted_idx = np.argsort(xgb_importance)[::-1]

for i in sorted_idx:
    print(f"{feature_names[i]}: {xgb_importance[i]:.4f}")

# -----------------------------
# SHAP (only for XGBoost)
# -----------------------------
print("\nGenerating SHAP plots...")

explainer = shap.TreeExplainer(xgb_model)

# Use smaller sample to avoid crash
X_sample = X_test.iloc[:50]

shap_values = explainer.shap_values(X_sample)

# Summary plot
shap.summary_plot(shap_values, X_sample, feature_names=feature_names)

# Bar plot
shap.summary_plot(shap_values, X_sample, feature_names=feature_names, plot_type="bar")

# Force plot (single prediction)
shap.force_plot(
    explainer.expected_value,
    shap_values[0],
    X_sample.iloc[0],
    feature_names=feature_names,
    matplotlib=True
)

# -----------------------------
# Save Model
# -----------------------------
joblib.dump(xgb_model, "xgb_model.pkl")
joblib.dump(scaler, "scaler.pkl")

print("\n✅ Model and scaler saved successfully!")