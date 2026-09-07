import streamlit as st
import pandas as pd
import joblib

model = joblib.load("churn_prediction_model.pkl")

st.title("Customer Churn Prediction")
st.write("Enter customer details below to estimate churn risk.")

age = st.number_input("Age", min_value=18, max_value=74, value=30)
tenure_months = st.number_input("Tenure (months)", min_value=1, max_value=119, value=24)
monthly_charges = st.number_input("Monthly charges", min_value=10.0, max_value=160.0, value=65.0)
num_products = st.number_input("Number of products", min_value=1, max_value=4, value=2)
credit_score = st.number_input("Credit score", min_value=300, max_value=850, value=650)
support_calls = st.number_input("Support calls", min_value=0, value=1)
complaints = st.selectbox("Complaints", ["No", "Yes"])
online_banking_active = st.selectbox("Online banking active", ["Yes", "No"])
missed_payments = st.number_input("Missed payments", min_value=0, value=0)
if st.button("Predict churn"):

    customer = pd.DataFrame([{
        "age": age,
        "tenure_months": tenure_months,
        "monthly_charges": monthly_charges,
        "num_products": num_products,
        "credit_score": credit_score,
        "support_calls": support_calls,
        "complaints": 1 if complaints == "Yes" else 0,
        "online_banking_active": 1 if online_banking_active == "Yes" else 0,
        "missed_payments": missed_payments
    }])

    churn_probability = model.predict_proba(customer)[0][1]

    st.metric("Churn probability", f"{churn_probability:.1%}")

    if churn_probability >= 0.7:
        st.error("High churn risk")
    elif churn_probability >= 0.4:
        st.warning("Medium churn risk")
    else:
        st.success("Low churn risk")
        
    risk_indicators = []

    if monthly_charges >= 100:
        risk_indicators.append("High monthly charges")

    if support_calls >= 4:
        risk_indicators.append("Frequent support calls")

    if complaints == "Yes":
        risk_indicators.append("Customer has complaints")

    if online_banking_active == "No":
        risk_indicators.append("Online banking is inactive")

    if missed_payments >= 2:
        risk_indicators.append("Multiple missed payments")

    if credit_score < 500:
        risk_indicators.append("Low credit score")

    if tenure_months < 12:
        risk_indicators.append("Short customer tenure")

    if risk_indicators:
        st.subheader("Risk indicators")

        for indicator in risk_indicators:
            st.write(f"- {indicator}")

        st.caption(
            "Risk indicators are simple rule-based flags added for explanation. "
            "They are not direct explanations of the machine learning model."
        )