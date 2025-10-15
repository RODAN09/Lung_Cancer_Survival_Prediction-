# app.py
import streamlit as st
import pandas as pd
import joblib
import datetime

# =============================
# Load Model
# =============================
model = joblib.load("lung_cancer_model_full.pkl")

# =============================
# Streamlit Page Config
# =============================
st.set_page_config(
    page_title="🫁 Lung Cancer Survival Prediction",
    layout="wide",
    page_icon="🫁"
)

# =============================
# Custom CSS for Modern Look
# =============================
st.markdown("""
<style>
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        font-size: 16px;
        height: 50px;
        width: 200px;
        border-radius: 10px;
    }
    .stProgress > div > div {
        background-color: #4CAF50;
    }
</style>
""", unsafe_allow_html=True)

# =============================
# Title and Description
# =============================
st.title("🫁 Lung Cancer Survival Prediction")
st.markdown("Predict the survival probability of a patient using medical data. Fill in the details below:")

# =============================
# 1️⃣ Patient Info
# =============================
st.header("🧍‍♂️ Patient Information")
col1, col2, col3 = st.columns(3)
with col1:
    age = st.number_input("Age", min_value=1, max_value=120, value=50)
with col2:
    gender = st.selectbox("Gender", ["Male", "Female"])
with col3:
    country = st.text_input("Country", "India")

# =============================
# 2️⃣ Medical Info
# =============================
st.header("🧬 Medical History")
col1, col2, col3, col4 = st.columns(4)
with col1:
    cancer_stage = st.selectbox("Cancer Stage", ["Stage I", "Stage II", "Stage III", "Stage IV"])
with col2:
    family_history = st.selectbox("Family History of Cancer", ["Yes", "No"])
with col3:
    smoking_status = st.selectbox("Smoking Status", ["Never Smoked", "Former Smoker", "Current Smoker", "Passive Smoker"])
with col4:
    bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=22.5)

col1, col2, col3 = st.columns(3)
with col1:
    cholesterol_level = st.number_input("Cholesterol Level", min_value=100, max_value=400, value=180)
with col2:
    hypertension = st.selectbox("Hypertension", ["Yes", "No"])
with col3:
    asthma = st.selectbox("Asthma", ["Yes", "No"])

col1, col2 = st.columns(2)
with col1:
    cirrhosis = st.selectbox("Cirrhosis", ["Yes", "No"])
with col2:
    other_cancer = st.selectbox("Other Cancer History", ["Yes", "No"])

# =============================
# 3️⃣ Treatment Info
# =============================
st.header("💊 Treatment")
col1, col2 = st.columns(2)
with col1:
    treatment_type = st.selectbox("Treatment Type", ["Surgery", "Chemotherapy", "Radiation", "Combined"])
with col2:
    start_date = st.date_input("Diagnosis Date", datetime.date(2022, 1, 1))
end_date = st.date_input("End of Treatment Date", datetime.date(2022, 6, 1))
treatment_duration_days = (end_date - start_date).days

# Additional features from training
diagnosis_year = start_date.year
diagnosis_month = start_date.month

# =============================
# Mapping for Categorical Features
# =============================
stage_mapping = {
    "Stage I": 1,
    "Stage II": 2,
    "Stage III": 3,
    "Stage IV": 4
}

input_data = pd.DataFrame({
    "age": [age],
    "gender": [1 if gender == "Male" else 0],
    "country": [0],  # placeholder
    "cancer_stage": [stage_mapping[cancer_stage]],
    "family_history": [1 if family_history == "Yes" else 0],
    "smoking_status": [0 if smoking_status == "Never Smoked" else 1],
    "bmi": [bmi],
    "cholesterol_level": [cholesterol_level],
    "hypertension": [1 if hypertension == "Yes" else 0],
    "asthma": [1 if asthma == "Yes" else 0],
    "cirrhosis": [1 if cirrhosis == "Yes" else 0],
    "other_cancer": [1 if other_cancer == "Yes" else 0],
    "treatment_type": [0 if treatment_type == "Surgery" else 1],
    "treatment_duration_days": [treatment_duration_days],
    "diagnosis_year": [diagnosis_year],
    "diagnosis_month": [diagnosis_month]
})

# Ensure numeric
input_data = input_data.astype(float)

# =============================
# 4️⃣ Prediction Button
# =============================
if st.button("🔍 Predict Survival"):
    try:
        prediction = model.predict(input_data)[0]
        prob = model.predict_proba(input_data)[0][1]

        st.markdown("### 🩺 Prediction Result")
        if prediction == 1:
            st.success(f"✅ The patient is likely to **SURVIVE**.")
        else:
            st.error(f"⚠️ The patient is **NOT likely to survive**.")

        # Probability bar
        st.markdown("#### 🔹 Survival Probability")
        st.progress(int(prob*100))

    except Exception as e:
        st.error(f"❌ Prediction failed: {e}")
        st.write("Input DataFrame:")
        st.dataframe(input_data)
        st.write(input_data.dtypes)
