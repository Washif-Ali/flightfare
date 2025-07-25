import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load pre-trained model (RandomForest or any)
model, expected_columns = joblib.load('flight_fare_model.pkl')

st.title("Flight Fare Prediction")

# Realistic options based on your dataset
airlines = ['SpiceJet', 'AirAsia', 'Vistara']
source_cities = ['Delhi', 'Chennai', 'Kolkata']
destination_cities = ['Mumbai', 'Hyderabad', 'Bangalore']
departure_times = ['Morning', 'Evening', 'Early_Morning']
arrival_times = ['Morning', 'Evening', 'Night']
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
    .stApp {
        background-color: #000000;
        background-size: cover;
        color: white;
    }
    </style>
    """,
    unsafe_allow_html=True
)
