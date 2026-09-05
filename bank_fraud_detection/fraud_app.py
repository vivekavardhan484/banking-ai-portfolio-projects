import streamlit as st
import pandas as pd
import joblib
model = joblib.load("bank_fraud_detection/fraud_detection_model.pkl")

st.set_page_config(
    page_title="Fraud Detection",
    page_icon="🏦"
)

st.title("Bank Fraud Detection")
st.write("Enter the transaction details below to check the fraud risk.")
st.caption(
    "Demo project built with synthetic transaction data. "
    "This is not a production banking fraud system."
)
amount = st.number_input(
    "Transaction amount (£)",
    min_value=0.0,
    value=100.0
)

transaction_hour = st.slider(
    "Transaction hour",
    0,
    23,
    12
)

account_age_days = st.number_input(
    "Account age in days",
    min_value=1,
    value=365
)

num_prev_transactions = st.number_input(
    "Number of previous transactions",
    min_value=0,
    value=10
)

is_international = st.selectbox(
    "International transaction?",
    ["No", "Yes"]
)

device_change = st.selectbox(
    "New or changed device?",
    ["No", "Yes"]
)

failed_login_attempts = st.number_input(
    "Failed login attempts",
    min_value=0,
    value=0
)
if st.button("Check transaction"):

    transaction = pd.DataFrame([{
        "amount": amount,
        "transaction_hour": transaction_hour,
        "account_age_days": account_age_days,
        "num_prev_transactions": num_prev_transactions,
        "is_international": 1 if is_international == "Yes" else 0,
        "device_change": 1 if device_change == "Yes" else 0,
        "failed_login_attempts": failed_login_attempts
    }])

    fraud_probability = model.predict_proba(transaction)[0][1]

    st.write(f"Fraud probability: {fraud_probability:.1%}")
    if fraud_probability >= 0.7:
        st.error("High risk transaction")

    elif fraud_probability >= 0.4:
        st.warning("Medium risk transaction")

    else:
        st.success("Low risk transaction")
    risk_indicators = []

    if amount >= 2000:
        risk_indicators.append("High transaction amount")

    if transaction_hour <= 4:
        risk_indicators.append("Transaction at an unusual hour")

    if account_age_days < 60:
        risk_indicators.append("New account")

    if is_international == "Yes":
        risk_indicators.append("International transaction")

    if device_change == "Yes":
        risk_indicators.append("New or changed device")

    if failed_login_attempts >= 3:
        risk_indicators.append("Multiple failed login attempts")

    if risk_indicators:
        st.subheader("Risk indicators")

        for indicator in risk_indicators:
            st.write(f"- {indicator}")    