import streamlit as st
import joblib
import numpy as np
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model = joblib.load(os.path.join(BASE_DIR, "credit_model.pkl"))
scaler = joblib.load(os.path.join(BASE_DIR, "scaler.pkl"))

st.title("💳 Loan Approval Prediction System")

st.write("Enter applicant details:")

income = st.number_input("Annual Income (₹)", min_value=10000, max_value=200000, step=1000)
credit_score = st.number_input("Credit Score", min_value=300, max_value=900)
debt_ratio = st.slider("Debt Ratio", 0.0, 1.0, step=0.01)
employment_years = st.number_input("Employment Years", min_value=0, max_value=40)
loan_amount = st.number_input("Loan Amount (₹)", min_value=10000, max_value=200000, step=1000)

# Adjustable threshold (VERY IMPORTANT FOR WORKSHOP)
threshold = st.slider("Decision Threshold", 0.1, 0.9, 0.4, step=0.005)

if st.button("Predict Loan Decision"):

    features = np.array([[income, credit_score, debt_ratio, employment_years, loan_amount]])
    features_scaled = scaler.transform(features)

    probability = model.predict_proba(features_scaled)[0][1]

    st.subheader("Result:")

    st.write(f"Risk Probability: **{probability:.2f}**")

    # Three-level decision logic
    if probability > threshold + 0.15:
        st.error(" Loan Likely to be Rejected (High Risk)")
    elif probability > threshold:
        st.warning(" Borderline Case – Manual Review Recommended")
    else:
        st.success(" Loan Likely to be Approved (Low Risk)")

    # Visual Risk Bar
    st.progress(int(probability * 100))