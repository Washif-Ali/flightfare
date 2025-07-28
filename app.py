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
import base64
import requests

image_url = "https://raw.githubusercontent.com/username/repo/branch/path/to/your-image.jpg"

response = requests.get(image_url)
response.raise_for_status()  # to catch errors

bg_image_base64 = base64.b64encode(response.content).decode()

st.markdown(
    f"""
    <style>
    .stApp {{
        position: relative;
        min-height: 100vh;
        overflow: hidden;
    }}

    .stApp::before {{
        content: "";
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background-image: url("data:image/png;base64,{bg_image_base64}");
        background-size: cover;
        background-position: center;
        z-index: -2;
    }}

    .stApp::after {{
        content: "";
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background-color: rgba(0, 0, 0, 0.6); /* Dark overlay */
        z-index: -1;
    }}

    h1, h3 {{
        color: #ffffff;
    }}

    .stButton>button {{
        background-color: #ff4b4b;
        color: white;
        padding: 10px 24px;
        border-radius: 8px;
        font-size: 16px;
        font-weight: bold;
        border: none;
    }}

    .stButton>button:hover {{
        background-color: #ff6b6b;
        color: black;
    }}

    .stSlider .st-cq {{
        color: #ff4b4b;
    }}

    .st-expander, .stSelectbox, .stSlider {{
        background-color: rgba(0, 0, 0, 0.7);
        border-radius: 10px;
        padding: 1em;
    }}

    .stSelectbox>div>div {{
        background-color: #222;
        color: white;
    }}

    .stTextInput>div>div>input {{
        color: white;
    }}
    </style>
    """,
    unsafe_allow_html=True
)

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

