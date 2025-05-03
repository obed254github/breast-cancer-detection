import streamlit as st
import pandas as pd
import joblib
import base64
import time


# Load model and column names
model = joblib.load("utils/cat_boost_model.pkl")
columns = joblib.load("utils/columns.pkl")

st.title("🧬 Breast Cancer Predictor")

st.write(
    "Input the details below about the lump to make predictions whether the tumor is **benign** or **malignant**."
)

# Create a dictionary to hold user inputs
input_data = {}

# Create text inputs with placeholders for each feature
for col in columns:
    value = st.text_input(f"{col}", placeholder="Enter a numeric value")
    try:
        input_data[col] = float(value)
    except:
        input_data[col] = None
if st.button("Predict Diagnosis"):
    # Check for missing values
    if None in input_data.values():
        st.error("Please fill in all fields with valid numeric values.")
    else:
        # creating a placeholder for holding animation
        loading = st.empty()
        # displaying centered gif via custom HTML
        loading.markdown(
            """
        <div style="
            position: fixed;
            top: 0; left: 0; right: 0; bottom: 0;
            display: flex;
            justify-content: center;
            align-items: center;
            background-color: rgba(0,0,0,0);
            z-index: 9999;">
            <img src="https://media.tenor.com/1tPR5fhN_AsAAAAm/world-cancer-day-survivor.webp" width="150">
        </div>
        """,
            unsafe_allow_html=True,
        )
        # Simulate loading
        time.sleep(2)

        # Clear the loading animation
        loading.empty()

        input_df = pd.DataFrame([input_data])
        prediction = model.predict(input_df)[0]
        label = "🟢 Benign" if prediction == 0 else "🔴 Malignant"
        st.success(f"Diagnosis: **{label}**")
