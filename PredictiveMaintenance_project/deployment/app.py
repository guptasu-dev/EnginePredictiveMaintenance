import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib

# Download and load the model
model_path = hf_hub_download(repo_id="Money2277/Engine-Predictive-Maintenance", filename="best_engine_maintenance_model.joblib")
model = joblib.load(model_path)

# Streamlit UI for Engine Predictive Maintenance
st.title("Engine Predictive Maintenance App")
st.write("""
This application predicts whether an engine is likely to be in a faulty condition based on its
sensor readings. Please enter the sensor data below to get a prediction.
""")

# User input
engine_rpm = st.number_input("Engine RPM", min_value=0.0, max_value=3000.0, value=750.0, step=1.0)
lub_oil_pressure = st.number_input("Lub Oil Pressure (bar)", min_value=0.0, max_value=10.0, value=3.3, step=0.1)
fuel_pressure = st.number_input("Fuel Pressure (bar)", min_value=0.0, max_value=25.0, value=6.6, step=0.1)
coolant_pressure = st.number_input("Coolant Pressure (bar)", min_value=0.0, max_value=10.0, value=2.3, step=0.1)
lub_oil_temp = st.number_input("Lub Oil Temperature (°C)", min_value=50.0, max_value=100.0, value=77.5, step=0.1)
coolant_temp = st.number_input("Coolant Temperature (°C)", min_value=50.0, max_value=100.0, value=78.0, step=0.1)

# Assemble input into DataFrame
input_data = pd.DataFrame([{
    'Engine rpm': engine_rpm,
    'Lub oil pressure': lub_oil_pressure,
    'Fuel pressure': fuel_pressure,
    'Coolant pressure': coolant_pressure,
    'lub oil temp': lub_oil_temp,
    'Coolant temp': coolant_temp
}])


if st.button("Predict Engine Condition"):
    prediction = model.predict(input_data)[0]
    result = "Faulty" if prediction == 1 else "Normal"
    st.subheader("Prediction Result:")
    if prediction == 1:
        st.error(f"The model predicts: **{result}**")
    else:
        st.success(f"The model predicts: **{result}**")
