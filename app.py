import streamlit as st
import pandas as pd
import joblib


rf = joblib.load("models/random_forest.pkl")
xgb = joblib.load("models/xgboost.pkl")

st.set_page_config( page_title="Cardioso", page_icon="❤️")
st.markdown(""" <style> @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap'); 
            html, body, [class*="css"] { 
            font-family: 'Inter', sans-serif; } 
            .stApp {
            background-color: #5A0F14;
            }
            h1 { font-weight: 700; } 
            button { font-weight: 600 !important; } 
            </style> """, unsafe_allow_html=True)
st.title("Cardioso")
st.write("Heart Disease Prediction using Random Forest and XGBoost")


st.header("Patient Information")


age = st.number_input(
    "Age",
    min_value=1,
    max_value=100,
    value=50
)

sex = st.selectbox(
    "Sex",
    ["Male", "Female"]
)

cp = st.selectbox(
    "Chest Pain Type",
    [
        "typical angina",
        "atypical angina",
        "non-anginal",
        "asymptomatic"
    ]
)

trestbps = st.number_input(
    "Resting Blood Pressure",
    min_value=50,
    max_value=250,
    value=120
)

chol = st.number_input(
    "Cholesterol",
    min_value=50,
    max_value=700,
    value=200
)

fbs = st.selectbox(
    "Fasting Blood Sugar > 120?",
    [False, True]
)

restecg = st.selectbox(
    "Resting ECG",
    [
        "normal",
        "lv hypertrophy",
        "st-t abnormality"
    ]
)

thalch = st.number_input(
    "Maximum Heart Rate",
    min_value=50,
    max_value=250,
    value=150
)

exang = st.selectbox(
    "Exercise-Induced Angina?",
    [False, True]
)

oldpeak = st.number_input(
    "Oldpeak",
    min_value=-5.0,
    max_value=10.0,
    value=0.0
)

slope = st.selectbox(
    "Slope",
    [
        "upsloping",
        "flat",
        "downsloping"
    ]
)

ca = st.selectbox(
    "Number of Major Vessels",
    [0, 1, 2, 3]
)

thal = st.selectbox(
    "Thalassemia",
    [
        "normal",
        "fixed defect",
        "reversable defect"
    ]
)


if st.button("Predict"):

    data = pd.DataFrame([{
        "age": age,
        "sex": sex,
        "cp": cp,
        "trestbps": trestbps,
        "chol": chol,
        "fbs": fbs,
        "restecg": restecg,
        "thalch": thalch,
        "exang": exang,
        "oldpeak": oldpeak,
        "slope": slope,
        "ca": ca,
        "thal": thal
    }])

    rf_p = rf.predict_proba(data)[0][1]
    xgb_p = xgb.predict_proba(data)[0][1]

    st.subheader("Results")

    col1, col2 = st.columns(2)

    with col1:
        st.write("### Random Forest")
        st.write(f"Probability: **{rf_p:.2%}**")

        if rf_p >= 0.5:
            st.error("Predicted: Disease")
        else:
            st.success("Predicted: No Disease")

    with col2:
        st.write("### XGBoost")
        st.write(f"Probability: **{xgb_p:.2%}**")

        if xgb_p >= 0.5:
            st.error("Predicted: Disease")
        else:
            st.success("Predicted: No Disease")


st.warning(
    "This prediction is not a medical diagnosis."
)