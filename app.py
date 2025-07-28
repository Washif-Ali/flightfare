import streamlit as st
import pandas as pd
import joblib
import numpy as np
import gdown
import os

# Google Drive model URL
file_id = "1KwIruPv9z2U_T86Lpv1KNkmfcUOKyKNj"
url = f"https://drive.google.com/uc?id={file_id}"
model_path = "flight_fare_model_rf.pkl"

# Download the model if needed
if not os.path.exists(model_path):
    gdown.download(url, model_path, quiet=False)

# Load model and expected columns with caching
@st.cache_resource
def load_model():
    return joblib.load(model_path)

model, expected_columns = load_model()

# --- UI Styling ---
st.markdown("""
    <style>
    body {
        background-color: #000000;
        color: white;
    }
    .stApp {
        background-color: #000000;
        color: white;
        font-family: 'Segoe UI', sans-serif;
    }
    h1, h3 {
        color: #00c2ff;
    }
    .stButton>button {
        background-color: #00c2ff;
        color: white;
        padding: 10px 24px;
        border-radius: 8px;
        font-size: 16px;
        font-weight: bold;
    }
    .stSlider .st-cq {
        color: #00c2ff;
    }
    </style>
""", unsafe_allow_html=True)

# --- App Title ---
st.title("✈️ Flight Fare Prediction")
st.subheader("🧳 Plan smarter. Pay less.")

# --- Options ---
airlines = ['SpiceJet', 'AirAsia', 'Vistara', 'GO_FIRST', 'Indigo', 'Air_India']
source_cities = ['Delhi', 'Mumbai', 'Bangalore', 'Kolkata', 'Hyderabad', 'Chennai']
destination_cities = ['Mumbai', 'Bangalore', 'Kolkata', 'Hyderabad', 'Chennai', 'Delhi']
departure_times = ['Early_Morning', 'Morning', 'Afternoon', 'Evening', 'Night', 'Late_Night']
arrival_times = ['Early_Morning', 'Morning', 'Afternoon', 'Evening', 'Night', 'Late_Night']
stops = ['zero', 'one', 'two_or_more']
classes = ['Economy', 'Business']

# --- Input Section ---
with st.expander("🧩 Enter Flight Details", expanded=True):
    col1, col2 = st.columns(2)
    with col1:
        airline = st.selectbox("✈️ Airline", airlines)
        source = st.selectbox("📍 Source City", source_cities)
        destination = st.selectbox("🏁 Destination City", destination_cities)
        stop_count = st.selectbox("🔁 Number of Stops", stops)
    with col2:
        travel_class = st.selectbox("🎟️ Class", classes)
        departure = st.selectbox("🕓 Departure Time", departure_times)
        arrival = st.selectbox("🕗 Arrival Time", arrival_times)
        duration = st.slider("⏱️ Flight Duration (hrs)", 1.0, 15.0, 2.0)
        days_left = st.slider("📅 Days Until Departure", 1, 60, 30)

# --- Prediction ---
if st.button("🚀 Predict Flight Fare"):
    with st.spinner("Predicting your fare..."):
        input_df = pd.DataFrame([{
            'airline': airline,
            'source_city': source,
            'destination_city': destination,
            'departure_time': departure,
            'arrival_time': arrival,
            'stops': stop_count,
            'class': travel_class,
            'duration': duration,
            'days_left': days_left
        }])

        input_df = pd.get_dummies(input_df)
        input_df = input_df.reindex(columns=expected_columns, fill_value=0)

        predicted_fare = model.predict(input_df)[0]

        st.success(f"✅ Estimated Fare: ₹{predicted_fare:,.2f}")

