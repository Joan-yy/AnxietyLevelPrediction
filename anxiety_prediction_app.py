import streamlit as st
import numpy as np
import joblib
import pandas as pd

# Load model and feature names
model = joblib.load("xgb_classifier_model.pkl")
feature_names = joblib.load("xgb_classifier_model_features.pkl")

# Page config
st.set_page_config(page_title="Anxiety Level Predictor", layout="wide")

# Custom styling
st.markdown("""
    <style>
    .main {
        background-color: #f4f6f7;
    }
    .stButton>button {
    
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
        padding: 10px 16px;
        border-radius: 8px;
    }
    </style>
""", unsafe_allow_html=True)

# Title
st.title("🧠 Anxiety Level Predictor")
st.write("Fill in the following details to get a prediction of your anxiety category.")

# --- Section: Basic Information ---
st.subheader("🔍 Basic Information")
col1, spacer, col2 = st.columns([1, 0.15, 1])
with col1:
    age = st.number_input("Age", min_value=10, max_value=100, value=25)
    gender = st.selectbox("Gender", ["Male", "Female"])
with col2:
    occupation = st.selectbox("Occupation", ["Student", "Engineer", "Doctor", "Teacher", "Others"])

# --- Section: Lifestyle & Habits ---
st.subheader("🛌 Lifestyle & Habits")
col1, spacer, col2 = st.columns([1, 0.15, 1])
with col1:
    sleep_hours = st.slider("Sleep Hours (per day)", 0.0, 12.0, 7.0)
    caffeine_intake = st.slider("Caffeine Intake (mg/day)", 0, 1000, 0)
    smoking = st.selectbox("Do you smoke?", ["No", "Yes"])
with col2:
    physical_activity = st.slider("Physical Activity (hours/week)", 0.0, 40.0, 0.0)
    alcohol = st.slider("Alcohol Consumption (drinks/week)", 0, 30, 0)
    diet = st.slider("Diet Quality (1-10)", 1, 10, 5)

# --- Section: Medical & Family History ---
st.subheader("🧬 Medical & Family History")
col1, spacer, col2 = st.columns([1, 0.15, 1])
with col1:
    family_history = st.selectbox("Family History of Anxiety?", ["No", "Yes"])
    therapy = st.slider("Therapy Sessions (per month)", 0, 20, 0)
with col2:
    medication = st.selectbox("Are you on medication?", ["No", "Yes"])

# --- Section: Psychological & Physical State ---
st.subheader("🧠 Psychological & Physical State")
col1, spacer, col2 = st.columns([1, 0.1, 1])
with col1:
    stress = st.slider("Stress Level (1-10)", 1, 10, 5)
    heart_rate = st.slider("Heart Rate (beats per minute)", 40, 140, 80)
with col2:
    breathing_rate = st.slider("Breathing Rate (breaths per minute)", 10, 40, 16)
    sweating = st.slider("Sweating Level (1-5)", 1, 5, 2)

dizziness = st.selectbox("Do you experience dizziness?", ["No", "Yes"])

# --- Section: Life Events ---
st.subheader("⚠️ Life Events")
life_event = st.selectbox("Recent Major Life Event?", ["No", "Yes"])

# --- Input Processing ---
gender = 1 if gender == "Male" else 0
smoking = 1 if smoking == "Yes" else 0
family_history = 1 if family_history == "Yes" else 0
dizziness = 1 if dizziness == "Yes" else 0
medication = 1 if medication == "Yes" else 0
life_event = 1 if life_event == "Yes" else 0

# One-hot encode occupation
occupation_columns = [col for col in feature_names if "Occupation_" in col]
occupation_data = [1 if f"Occupation_{occupation}" == col else 0 for col in occupation_columns]

# Input vector
input_data = [
    age, gender, sleep_hours, physical_activity, caffeine_intake, alcohol,
    smoking, family_history, stress, heart_rate, breathing_rate, sweating,
    dizziness, medication, therapy, life_event, diet
] + occupation_data

input_df = pd.DataFrame([input_data], columns=feature_names)

# --- Prediction ---
st.markdown("---")
if st.button("🎯 Predict Anxiety Category"):
    prediction = model.predict(input_df)[0]

    st.subheader("🧾 Predicted Anxiety Category:")

    if prediction == 0:
        st.success("🟢 **Low Anxiety**\n\nGreat! Your anxiety level appears to be low. Keep maintaining a healthy lifestyle and positive habits.")
        st.markdown("✅ Tips:\n- Stay physically active\n- Maintain a regular sleep schedule\n- Keep social connections strong")
    
    elif prediction == 1:
        st.warning("🟡 **Moderate Anxiety**\n\nYou might be experiencing some moderate anxiety. It’s helpful to monitor how you're feeling and consider mild interventions.")
        st.markdown("⚠️ Suggestions:\n- Practice mindfulness or breathing exercises\n- Take regular breaks\n- Try journaling your thoughts")
    
    else:
        st.error("🔴 **High Anxiety**\n\nYou may be experiencing high levels of anxiety. Consider seeking support from a healthcare professional.")
        st.markdown("🆘 Recommendations:\n- Reach out to a counselor or therapist\n- Prioritize rest and self-care\n- Talk to someone you trust")

    st.markdown("---")
    st.caption("Note: This prediction is not a medical diagnosis. For serious concerns, consult a licensed professional.")
