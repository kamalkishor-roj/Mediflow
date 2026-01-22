import streamlit as st
import requests
import os

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000/predict")

st.set_page_config(
    page_title="MediFlow – Sepsis Risk Dashboard",
    layout="centered"
)

st.title("🩺 MediFlow: Sepsis Risk Predictor")
st.markdown("Enter patient vitals to estimate sepsis risk.")

# -----------------------------
# Input form
# -----------------------------
with st.form("patient_form"):
    HR = st.number_input("Heart Rate (bpm)", min_value=30, max_value=200, value=96)
    O2Sat = st.number_input("Oxygen Saturation (%)", min_value=50, max_value=100, value=98)
    Temp = st.number_input("Temperature (°C)", min_value=34.0, max_value=42.0, value=38.1)
    SBP = st.number_input("Systolic BP (mm Hg)", min_value=70, max_value=250, value=120)
    MAP = st.number_input("Mean Arterial Pressure (mm Hg)", min_value=40, max_value=200, value=75)
    DBP = st.number_input("Diastolic BP (mm Hg)", min_value=40, max_value=150, value=70)
    Resp = st.number_input("Respiration Rate (breaths/min)", min_value=5, max_value=60, value=22)
    EtCO2 = st.number_input("EtCO2 (mm Hg)", min_value=10, max_value=60, value=30)

    submitted = st.form_submit_button("🔍 Predict Sepsis Risk")

# -----------------------------
# Prediction
# -----------------------------
if submitted:
    payload = {
        "features": {
            "HR": HR,
            "O2Sat": O2Sat,
            "Temp": Temp,
            "SBP": SBP,
            "MAP": MAP,
            "DBP": DBP,
            "Resp": Resp,
            "EtCO2": EtCO2
        }
    }

    try:
        response = requests.post(API_URL, json=payload)

        if response.status_code == 200:
            risk = response.json()["sepsis_risk_score"]

            st.success(f"🧠 Estimated Sepsis Risk: **{risk}%**")

            if risk < 10:
                st.info("Low risk — continue routine monitoring.")
            elif risk < 25:
                st.warning("Moderate risk — observe closely.")
            else:
                st.error("High risk — immediate clinical attention recommended.")

        else:
            st.error("API error. Please check the backend server.")

    except Exception as e:
        st.error(f"Could not connect to API: {e}")
