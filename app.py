import streamlit as st
import pandas as pd
import joblib
import numpy as np
import gdown
import os

# File ID from Google Drive
file_id = "1KwIruPv9z2U_T86Lpv1KNkmfcUOKyKNj"
url = f"https://drive.google.com/uc?id={file_id}"

# Local path to save model
model_path = "flight_fare_model_rf.pkl"

# Only download if not already downloaded
if not os.path.exists(model_path):
    gdown.download(url, model_path, quiet=False)

# Load the model
model, expected_columns = joblib.load(model_path)

st.title("Flight Fare Prediction")

# Realistic options based on your dataset
airlines = ['SpiceJet', 'AirAsia', 'Vistara', 'GO_FIRST', 'Indigo', 'Air_India']
source_cities = ['Delhi', 'Mumbai', 'Bangalore', 'Kolkata', 'Hyderabad', 'Chennai']
destination_cities = ['Mumbai', 'Bangalore', 'Kolkata', 'Hyderabad', 'Chennai', 'Delhi']
departure_times = ['Early_Morning', 'Morning', 'Afternoon', 'Evening', 'Night', 'Late_Night']
arrival_times = ['Early_Morning', 'Morning', 'Afternoon', 'Evening', 'Night', 'Late_Night']
stops = ['zero', 'one', 'two_or_more']
classes = ['Economy', 'Business']

# User inputs
airline = st.selectbox("Airline", airlines)
source = st.selectbox("Source City", source_cities)
destination = st.selectbox("Destination City", destination_cities)
departure = st.selectbox("Departure Time", departure_times)
arrival = st.selectbox("Arrival Time", arrival_times)
stop_count = st.selectbox("Number of Stops", stops)
travel_class = st.selectbox("Class", classes)
duration = st.slider("Flight Duration (hours)", 1.0, 15.0, 2.0)
days_left = st.slider("Days Left to Departure", 1, 60, 30)

# Prediction
if st.button("Predict Flight Fare"):

    # Create input DataFrame
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

    # One-hot encode to match training format
    input_df = pd.get_dummies(input_df)


    # Align columns with model input
    input_df = input_df.reindex(columns=expected_columns, fill_value=0)


    # Predict
    predicted_fare = model.predict(input_df)[0]
    st.success(f"Predicted Flight Fare: ₹{predicted_fare:,.2f}")

# Optional: Style
st.markdown(
    """
    <style>
    .stApp {{
        background-color: #000000;
        background-size: cover;
        color: white;
    }}
    </style>
    """,
    unsafe_allow_html=True
)
