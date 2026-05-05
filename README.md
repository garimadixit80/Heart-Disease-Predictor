# ❤️ Heart Disease Prediction System

A complete **Machine Learning-powered web application** that predicts the likelihood of heart disease based on patient health parameters. The system uses advanced models like XGBoost and provides real-time predictions through an interactive UI.

---

## 🚀 Live Demo

👉 **Try the app here:**
🔗 https://heart-disease-predictor2604.streamlit.app/

---

## 📌 Project Overview

This project aims to build an **end-to-end ML system** that:

* Predicts heart disease risk
* Compares multiple machine learning models
* Provides interpretability using SHAP
* Deploys a real-time prediction interface using Streamlit

---

## 🧠 Models Used

* Logistic Regression (Baseline)
* Random Forest (Robust & stable)
* XGBoost (High performance)

---

## 📊 Performance Summary

| Model               | Accuracy | Precision | Recall | F1 Score |
| ------------------- | -------- | --------- | ------ | -------- |
| Logistic Regression | 0.80     | 0.77      | 0.87   | 0.82     |
| Random Forest       | 0.89     | 0.88      | 0.92   | 0.90     |
| XGBoost             | 0.98     | 0.99      | 0.97   | 0.98     |

✅ Random Forest showed strong generalization with cross-validation (~92%)
⚠️ XGBoost achieved highest accuracy but may slightly overfit due to small dataset

---

## 🔍 Key Insights

* **Top predictive features:**

  * Thalassemia (`thal`)
  * Chest Pain Type (`cp`)
  * Number of vessels (`ca`)
  * Exercise-induced angina (`exang`)
  * ST depression (`oldpeak`)

* Tree-based models captured **non-linear relationships** better than Logistic Regression

* Focused on minimizing **false negatives**, which is critical in healthcare applications

---

## 📈 Model Explainability (SHAP)

* Integrated SHAP to explain model predictions
* Helps understand **why a prediction was made**
* Improves transparency in medical decision-making

---

## 💻 Tech Stack

* Python
* Pandas, NumPy
* Scikit-learn
* XGBoost
* SHAP
* Streamlit

---

## 🖥️ Features

* Interactive web UI using Streamlit
* Real-time heart disease prediction
* Multiple model comparison
* Feature importance analysis
* SHAP-based explainability

---

## ⚙️ Installation & Setup

### 1. Clone the repository

```bash
git clone YOUR_REPO_LINK
cd heart-disease-prediction
```

### 2. Create virtual environment

```bash
python -m venv venv
```

Activate:

```bash
venv\Scripts\activate   # Windows
source venv/bin/activate  # Mac/Linux
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the application

```bash
streamlit run app.py
```

---

## 📁 Project Structure

```
heart-disease-prediction/
│
├── app.py              # Streamlit UI
├── train.py            # Model training script
├── heart.csv           # Dataset
├── xgb_model.pkl       # Saved model
├── scaler.pkl          # Scaler
├── requirements.txt
└── README.md
```

---

## 🧪 Future Improvements

* Add ROC-AUC curve visualization
* Integrate SHAP directly into UI
* Deploy using Docker for scalability
* Use larger real-world datasets

---

## 🧠 Learnings

* Importance of avoiding data leakage
* Model evaluation beyond accuracy (Precision, Recall, F1)
* Trade-off between interpretability and performance
* Deployment of ML models as real-world applications

---

## 📬 Contact

If you found this useful or have suggestions, feel free to connect!

