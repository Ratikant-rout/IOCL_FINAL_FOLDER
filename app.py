import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

# App config
st.set_page_config(page_title="IOCL Failure Predictor", layout="wide")
st.title("🔧 IOCL Equipment Failure Prediction")
st.markdown("Enter equipment readings manually to get a failure prediction.")

# Equipment options
equipment = st.sidebar.selectbox("Select Equipment", ["Pump", "Compressor", "Heat Exchanger"])

# Feature sets
feature_sets = {
    "Pump": [
        "Inlet_Temperature(°C)",
        "Outlet_Temperature(°C)",
        "Pressure(bar)",
        "FlowRate(LPM)",
        "Vibration(mm/s)",
        "Differential_Pressure(bar)"
    ],
    "Compressor": [
        "Inlet_Temperature(°C)",
        "Outlet_Temperature(°C)",
        "Suction_Pressure(bar)",
        "Discharge_Pressure(bar)",
        "Vibration(mm/s)",
        "Differential_Pressure(bar)"
    ],
    "Heat Exchanger": [
        "Inlet_Temperature(°C)",
        "Outlet_Temperature(°C)",
        "FlowRate(LPM)",
        "Pressure(bar)",
        "Vibration(mm/s)",
        "Differential_Pressure(bar)"
    ]
}
feature_names = feature_sets[equipment]

# Load model and scaler
@st.cache_resource
def load_model_and_scaler(equipment):
    if equipment == "Pump":
        model = load_model("models/pump_model.h5", compile=False)
        df = pd.read_csv("data/PumpsFinal.csv")
    elif equipment == "Compressor":
        model = load_model("models/compressor_model.h5", compile=False)
        df = pd.read_csv("data/CompressorFinal.csv")
    else:
        model = load_model("models/heat_exchanger_model.h5", compile=False)
        df = pd.read_csv("data/HeatExchangerFinal.csv")
    
    scaler = MinMaxScaler()
    scaler.fit(df.drop(columns=["Failure"]))
    return model, scaler

model, scaler = load_model_and_scaler(equipment)

# Sidebar inputs
st.sidebar.header(f"📥 Enter {equipment} Parameters")
user_input = [st.sidebar.number_input(f"{feature}", min_value=0.0, step=0.1) for feature in feature_names]

# Prediction
if st.sidebar.button("🔍 Predict Failure"):
    X = np.array(user_input).reshape(1, -1)
    X_scaled = scaler.transform(X)
    prediction = model.predict(X_scaled)[0][0]
    failure_percent = prediction * 100

    # Interpretation
    st.subheader("📋 Parameter Interpretation")
    interpretations = []

    if user_input[4] > 7.0:
        interpretations.append("🔴 High vibration → Possible misalignment or mechanical wear.")
    if equipment == "Pump" and user_input[3] < 300:
        interpretations.append("🟠 Low flow rate → Possible clogging or blockage.")
    if equipment == "Compressor":
        if user_input[2] < 1.0:
            interpretations.append("🟠 Low suction pressure → Suction restriction or leak.")
        if user_input[3] > 10:
            interpretations.append("🟠 High discharge pressure → Discharge valve issue.")
    if user_input[5] > 1.5:
        interpretations.append("🟠 High differential pressure → Internal fouling or scaling.")
    if user_input[0] < 10 or user_input[0] > 90:
        interpretations.append("🟡 Abnormal inlet temperature → Risk of thermal stress.")
    if user_input[1] > 120:
        interpretations.append("🟡 High outlet temperature → Overheating risk.")

    if not interpretations:
        interpretations.append("✅ All parameters are within typical operating range.")

    for note in interpretations:
        st.write(note)

    # Plot inputs
    st.subheader("📊 Input Parameter Values")
    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(feature_names, user_input, color='skyblue')
    ax.set_ylabel("Sensor Value")
    ax.set_title("Entered Parameters")
    plt.xticks(rotation=30)

    for bar, val, name in zip(bars, user_input, feature_names):
        if name == "Vibration(mm/s)" and val > 7.0:
            bar.set_color("red")
        if name == "FlowRate(LPM)" and val < 300 and equipment == "Pump":
            bar.set_color("orange")
        if name == "Differential_Pressure(bar)" and val > 1.5:
            bar.set_color("orange")

    st.pyplot(fig)

else:
    st.info("👈 Enter parameters in the sidebar and click Predict.")
