# Predictive Customer Churn Prediction

## 📌 Project Overview

Customer churn prediction is the process of identifying customers who are likely to stop using a company's services.

This project uses Machine Learning to predict whether a customer is likely to churn based on demographic, service, contract, and billing information.

The final model is implemented using **XGBoost** and deployed as an interactive **Streamlit web application**.

---

## 🎯 Objectives

- Predict whether a customer is likely to churn.
- Identify important factors influencing customer churn.
- Compare different machine learning models.
- Evaluate model performance using multiple metrics.
- Use SHAP for model explainability.
- Deploy the final model through a Streamlit application.

---

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- XGBoost
- SHAP
- Joblib
- Streamlit
- Jupyter Notebook

---

## 🤖 Machine Learning Models

The project compares the following models:

1. Logistic Regression
2. Random Forest
3. XGBoost

After model evaluation and tuning, **XGBoost** was selected as the final model.

---

## 📊 Model Performance

The final tuned XGBoost model achieved:

| Metric | Score |
|---|---:|
| Accuracy | 0.8048 |
| Precision | 0.6713 |
| Recall | 0.5187 |
| F1 Score | 0.5852 |
| AUC-ROC | 0.8456 |

The ROC-AUC comparison showed that XGBoost achieved the highest AUC among the evaluated models.

---

## 🔍 Model Explainability

SHAP (SHapley Additive exPlanations) was used to understand how different features influence the XGBoost predictions.

The SHAP analysis helps identify important factors affecting customer churn, including features related to:

- Contract type
- Tenure
- Online security
- Internet service
- Tech support
- Total charges
- Monthly charges
- Payment method

---

## 🌐 Streamlit Application

The project includes an interactive Streamlit web application.

Users can enter customer information such as:

- Gender
- Partner
- Senior Citizen
- Dependents
- Tenure
- Phone Service
- Online Security
- Tech Support
- Multiple Lines
- Online Backup
- Streaming TV
- Streaming Movies
- Internet Service
- Device Protection
- Contract
- Payment Method
- Paperless Billing
- Monthly Charges
- Total Charges

The application then provides:

- Churn prediction
- Churn probability
- Risk level
- Customer summary

---

## 📁 Project Structure

```text
Predictive_Customer_Churn_Project/
│
├── data/
│   └── WA_Fn-UseC_-Telco-Customer-Churn
│
├── model/
│   ├── churn_xgb_model.pkl
│   ├── feature_names.pkl
│   └── preprocessor.pkl
│
├── notebooks/
│   └── customer_churn_eda.ipynb
│
├── reports/
│
├── screenshots/
│
├── streamlit_app/
│   └── app.py
│
├── tests/
│
├── .gitignore
├── README.md
└── requirements.txt