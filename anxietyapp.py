import streamlit as st
import numpy as np
import joblib
import pandas as pd

# Load model and feature names
model = joblib.load("model.pkl")
feature_names = joblib.load("model_features_lr.pkl")

# App title
st.title("Anxiety Level Predictor")
st.write("Fill in the following details to predict your anxiety category.")

# User input
age = st.number_input("Age", min_value=10, max_value=100, value=25)
gender = st.selectbox("Gender", ["Male", "Female"])
occupation = st.selectbox("Occupation", ["Student", "Engineer", "Doctor", "Teacher", "Others"])
sleep_hours = st.slider("Sleep Hours (per day)", 0.0, 12.0, 7.0)
physical_activity = st.slider("Physical Activity (hrs/week)", 0.0, 20.0, 3.0)
caffeine_intake = st.slider("Caffeine Intake (mg/day)", 0, 1000, 200)
alcohol = st.slider("Alcohol Consumption (drinks/week)", 0, 30, 0)
smoking = st.selectbox("Do you smoke?", ["No", "Yes"])
family_history = st.selectbox("Family History of Anxiety?", ["No", "Yes"])
stress = st.slider("Stress Level (1-10)", 1, 10, 5)
heart_rate = st.slider("Heart Rate (bpm)", 40, 140, 70)
breathing_rate = st.slider("Breathing Rate (breaths/min)", 10, 40, 16)
sweating = st.slider("Sweating Level (1-5)", 1, 5, 3)
dizziness = st.selectbox("Do you experience dizziness?", ["No", "Yes"])
medication = st.selectbox("Are you on medication?", ["No", "Yes"])
therapy = st.slider("Therapy Sessions (per month)", 0, 20, 0)
life_event = st.selectbox("Recent Major Life Event?", ["No", "Yes"])
diet = st.slider("Diet Quality (1-10)", 1, 10, 6)

# Convert categorical inputs
gender = 1 if gender == "Male" else 0
smoking = 1 if smoking == "Yes" else 0
family_history = 1 if family_history == "Yes" else 0
dizziness = 1 if dizziness == "Yes" else 0
medication = 1 if medication == "Yes" else 0
life_event = 1 if life_event == "Yes" else 0

# One-hot encode Occupation
occupation_columns = [col for col in feature_names if "Occupation_" in col]
occupation_data = [1 if f"Occupation_{occupation}" == col else 0 for col in occupation_columns]

# Full input in the correct order
input_data = [age, gender, sleep_hours, physical_activity, caffeine_intake, alcohol,
              smoking, family_history, stress, heart_rate, breathing_rate, sweating,
              dizziness, medication, therapy, life_event, diet] + occupation_data

# Ensure correct shape
input_df = pd.DataFrame([input_data], columns=feature_names)

# Predict button
if st.button("Predict"):
    prediction = model.predict(input_df)[0]
    label_map = {0: "Low", 1: "Medium", 2: "High"}
    st.subheader("Predicted Anxiety Category:")
    st.success(label_map[prediction])
