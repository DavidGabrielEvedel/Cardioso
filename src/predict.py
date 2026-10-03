import pandas as pd
import joblib

rf = joblib.load("models/random_forest.pkl")
xgb = joblib.load("models/xgboost.pkl")

age = int(input("Age: "))
sex = input("Sex (Male/Female): ")
cp = input("Chest pain type: ")
trestbps = float(input("Resting blood pressure: "))
chol = float(input("Cholesterol: "))
fbs = input("Fasting blood sugar (True/False): ") == "True"
restecg = input("Resting ECG: ")
thalch = float(input("Maximum heart rate: "))
exang = input("Exercise induced angina (True/False): ") == "True"
oldpeak = float(input("Oldpeak: "))
slope = input("Slope: ")
ca = float(input("Number of major vessels (0-3): "))
thal = input("Thal: ")

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

print("\n--- Prediction ---")

print("Random Forest probability:", round(rf_p * 100, 2), "%")
print("XGBoost probability:", round(xgb_p * 100, 2), "%")

print("\nRandom Forest:",
      "Disease" if rf_p >= 0.5 else "No Disease")

print("XGBoost:",
      "Disease" if xgb_p >= 0.5 else "No Disease")