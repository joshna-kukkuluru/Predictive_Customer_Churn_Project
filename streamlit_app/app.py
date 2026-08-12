import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)


# =========================================================
# LOAD SAVED MODEL, PREPROCESSOR AND FEATURE NAMES
# =========================================================

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR.parent / "model"

MODEL_PATH = MODEL_DIR / "churn_xgb_model.pkl"
PREPROCESSOR_PATH = MODEL_DIR / "preprocessor.pkl"
FEATURE_NAMES_PATH = MODEL_DIR / "feature_names.pkl"


@st.cache_resource
def load_artifacts():

    model = joblib.load(MODEL_PATH)
    preprocessor = joblib.load(PREPROCESSOR_PATH)
    feature_names = joblib.load(FEATURE_NAMES_PATH)

    return model, preprocessor, feature_names


try:
    model, preprocessor, feature_names = load_artifacts()

except Exception as e:

    st.error("❌ Model files could not be loaded.")

    st.write("Expected model files:")
    st.write(str(MODEL_DIR))

    st.exception(e)

    st.stop()


# =========================================================
# TITLE
# =========================================================

st.title("📊 Customer Churn Prediction")

st.write(
    "Predict whether a customer is likely to churn based on "
    "their demographic, service, contract and billing information."
)

st.divider()


# =========================================================
# CUSTOMER INFORMATION
# =========================================================

st.header("👤 Customer Information")

col1, col2, col3 = st.columns(3)

with col1:

    gender = st.selectbox(
        "Gender",
        ["Male", "Female"]
    )

    senior_citizen = st.selectbox(
        "Senior Citizen",
        [0, 1],
        format_func=lambda x: "Yes" if x == 1 else "No"
    )

with col2:

    partner = st.selectbox(
        "Partner",
        ["Yes", "No"]
    )

    dependents = st.selectbox(
        "Dependents",
        ["Yes", "No"]
    )

with col3:

    tenure = st.number_input(
        "Tenure (months)",
        min_value=0,
        max_value=100,
        value=12,
        step=1
    )


# =========================================================
# PHONE AND INTERNET SERVICES
# =========================================================

st.header("📱 Services")

col1, col2, col3 = st.columns(3)

with col1:

    phone_service = st.selectbox(
        "Phone Service",
        ["Yes", "No"]
    )

    multiple_lines = st.selectbox(
        "Multiple Lines",
        [
            "Yes",
            "No",
            "No phone service"
        ]
    )

    internet_service = st.selectbox(
        "Internet Service",
        [
            "DSL",
            "Fiber optic",
            "No"
        ]
    )

with col2:

    online_security = st.selectbox(
        "Online Security",
        [
            "Yes",
            "No",
            "No internet service"
        ]
    )

    online_backup = st.selectbox(
        "Online Backup",
        [
            "Yes",
            "No",
            "No internet service"
        ]
    )

    device_protection = st.selectbox(
        "Device Protection",
        [
            "Yes",
            "No",
            "No internet service"
        ]
    )

with col3:

    tech_support = st.selectbox(
        "Tech Support",
        [
            "Yes",
            "No",
            "No internet service"
        ]
    )

    streaming_tv = st.selectbox(
        "Streaming TV",
        [
            "Yes",
            "No",
            "No internet service"
        ]
    )

    streaming_movies = st.selectbox(
        "Streaming Movies",
        [
            "Yes",
            "No",
            "No internet service"
        ]
    )


# =========================================================
# CONTRACT AND BILLING
# =========================================================

st.header("💳 Contract & Billing")

col1, col2, col3 = st.columns(3)

with col1:

    contract = st.selectbox(
        "Contract",
        [
            "Month-to-month",
            "One year",
            "Two year"
        ]
    )

    paperless_billing = st.selectbox(
        "Paperless Billing",
        ["Yes", "No"]
    )

with col2:

    payment_method = st.selectbox(
        "Payment Method",
        [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)"
        ]
    )

with col3:

    monthly_charges = st.number_input(
        "Monthly Charges",
        min_value=0.0,
        max_value=1000.0,
        value=70.0,
        step=0.01
    )

    total_charges_input = st.number_input(
        "Total Charges",
        min_value=0.0,
        max_value=10000.0,
        value=840.0,
        step=0.01
    )


st.divider()


# =========================================================
# PREDICTION BUTTON
# =========================================================

predict_button = st.button(
    "🔮 Predict Customer Churn",
    type="primary",
    use_container_width=True
)


# =========================================================
# PREDICTION
# =========================================================

if predict_button:

    try:

        # -------------------------------------------------
        # Feature Engineering
        # Same logic as notebook
        # -------------------------------------------------

        # TotalChargesPerTenure
        if tenure > 0:

            total_charges_per_tenure = (
                total_charges_input / tenure
            )

        else:

            total_charges_per_tenure = monthly_charges


        # ServiceCount
        service_values = [
            phone_service,
            multiple_lines,
            internet_service,
            online_security,
            online_backup,
            device_protection,
            tech_support,
            streaming_tv,
            streaming_movies
        ]

        service_count = sum(
            value not in [
                "No",
                "No internet service",
                "No phone service"
            ]
            for value in service_values
        )


        # IsLongTermCustomer
        is_long_term_customer = (
            1 if tenure >= 24 else 0
        )


        # -------------------------------------------------
        # Create input DataFrame
        # Exact feature names used before preprocessing
        # -------------------------------------------------

        input_data = pd.DataFrame({

            "gender": [gender],

            "SeniorCitizen": [senior_citizen],

            "Partner": [partner],

            "Dependents": [dependents],

            "tenure": [tenure],

            "PhoneService": [phone_service],

            "MultipleLines": [multiple_lines],

            "InternetService": [internet_service],

            "OnlineSecurity": [online_security],

            "OnlineBackup": [online_backup],

            "DeviceProtection": [device_protection],

            "TechSupport": [tech_support],

            "StreamingTV": [streaming_tv],

            "StreamingMovies": [streaming_movies],

            "Contract": [contract],

            "PaperlessBilling": [paperless_billing],

            "PaymentMethod": [payment_method],

            "MonthlyCharges": [monthly_charges],

            "TotalCharges": [total_charges_input],

            "TotalChargesPerTenure": [
                total_charges_per_tenure
            ],

            "ServiceCount": [service_count],

            "IsLongTermCustomer": [
                is_long_term_customer
            ]
        })


        # -------------------------------------------------
        # Verify input columns
        # -------------------------------------------------

        expected_columns = list(
            preprocessor.feature_names_in_
        )

        if list(input_data.columns) != expected_columns:

            st.error(
                "❌ Input feature order does not match "
                "the training data."
            )

            st.write("Expected columns:")
            st.write(expected_columns)

            st.write("Received columns:")
            st.write(list(input_data.columns))

            st.stop()


        # -------------------------------------------------
        # Apply saved preprocessing pipeline
        # -------------------------------------------------

        processed_data = preprocessor.transform(
            input_data
        )


        # -------------------------------------------------
        # Verify processed feature count
        # -------------------------------------------------

        if processed_data.shape[1] != len(feature_names):

            st.error(
                "❌ Processed feature count does not match "
                "the saved model."
            )

            st.write(
                "Processed features:",
                processed_data.shape[1]
            )

            st.write(
                "Expected features:",
                len(feature_names)
            )

            st.stop()


        # -------------------------------------------------
        # Prediction
        # -------------------------------------------------

        prediction = model.predict(
            processed_data
        )[0]

        probability = model.predict_proba(
            processed_data
        )[0][1]


        # =================================================
        # DISPLAY RESULT
        # =================================================

        st.divider()

        st.header("📌 Prediction Result")

        result_col1, result_col2 = st.columns(2)


        with result_col1:

            if prediction == 1:

                st.error(
                    "⚠️ Customer is likely to CHURN"
                )

            else:

                st.success(
                    "✅ Customer is likely to STAY"
                )


        with result_col2:

            st.metric(
                "Churn Probability",
                f"{probability * 100:.2f}%"
            )


        # -------------------------------------------------
        # Risk Level
        # -------------------------------------------------

        if probability >= 0.70:

            risk_level = "🔴 High Risk"

        elif probability >= 0.40:

            risk_level = "🟠 Medium Risk"

        else:

            risk_level = "🟢 Low Risk"


        st.subheader("Risk Level")

        st.write(f"### {risk_level}")


        # -------------------------------------------------
        # Input Summary
        # -------------------------------------------------

        st.subheader("Customer Summary")

        summary_col1, summary_col2, summary_col3 = st.columns(3)

        with summary_col1:

            st.write(
                f"**Tenure:** {tenure} months"
            )

            st.write(
                f"**Contract:** {contract}"
            )

        with summary_col2:

            st.write(
                f"**Monthly Charges:** "
                f"${monthly_charges:.2f}"
            )

            st.write(
                f"**Internet Service:** "
                f"{internet_service}"
            )

        with summary_col3:

            st.write(
                f"**Payment Method:** "
                f"{payment_method}"
            )

            st.write(
                f"**Service Count:** "
                f"{service_count}"
            )


    except Exception as e:

        st.error(
            "❌ An error occurred while making the prediction."
        )

        st.exception(e)