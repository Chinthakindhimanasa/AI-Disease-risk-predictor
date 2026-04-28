import streamlit as st
import numpy as np
import joblib
import matplotlib.pyplot as plt
import pyttsx3

# ---------------- LOAD MODEL ---------------- #
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

# ---------------- VOICE ENGINE ---------------- #
engine = pyttsx3.init()

# ---------------- UI ---------------- #
st.title("🧠🏥 AI Disease Risk Prediction System")
st.write("Enter patient health details:")

# ---------------- INPUTS ---------------- #
preg = st.number_input("Pregnancies", min_value=0, value=1)
glucose = st.number_input("Glucose", min_value=0, value=120)
bp = st.number_input("Blood Pressure", min_value=0, value=70)
skin = st.number_input("Skin Thickness", min_value=0, value=20)
insulin = st.number_input("Insulin", min_value=0, value=80)
bmi = st.number_input("BMI", min_value=0.0, value=25.0)
dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, value=0.5)
age = st.number_input("Age", min_value=0, max_value=120, value=25)

# ---------------- SHOW INPUT ---------------- #
st.subheader("📋 Patient Data")
st.json({
    "Pregnancies": preg,
    "Glucose": glucose,
    "Blood Pressure": bp,
    "Skin Thickness": skin,
    "Insulin": insulin,
    "BMI": bmi,
    "Diabetes Pedigree Function": dpf,
    "Age": age
})

# ---------------- PREDICTION ---------------- #
if st.button("Predict Risk"):

    # Convert input
    input_data = np.array([[preg, glucose, bp, skin, insulin, bmi, dpf, age]])
    input_scaled = scaler.transform(input_data)

    # Predict
    prediction = model.predict(input_scaled)
    prob = model.predict_proba(input_scaled)

    # Result logic
    if prediction[0] == 1:
        st.error("🔴 High Risk of Diabetes")
        result_text = "High risk of diabetes"
    else:
        st.success("🟢 Low Risk of Diabetes")
        result_text = "Low risk of diabetes"

    st.write("📊 Probability:", prob)

    # ---------------- VOICE OUTPUT ---------------- #
    engine.say(result_text)
    engine.runAndWait()

    # ---------------- FEATURE IMPORTANCE ---------------- #
    st.subheader("🧠 Feature Importance")

    importance = model.feature_importances_

    features = [
        "Pregnancies", "Glucose", "Blood Pressure", "Skin Thickness",
        "Insulin", "BMI", "Diabetes Pedigree Function", "Age"
    ]

    fig, ax = plt.subplots()
    ax.barh(features, importance)
    ax.set_xlabel("Importance Score")

    st.pyplot(fig)