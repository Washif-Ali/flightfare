import streamlit as st
import pandas as pd
import numpy as np
import pydeck as pdk
import joblib
import gdown
import os

# --- MODEL LOADING ---
file_id = "1KwIruPv9z2U_T86Lpv1KNkmfcUOKyKNj"
url = f"https://drive.google.com/uc?id={file_id}"
model_path = "flight_fare_model_rf.pkl"

if not os.path.exists(model_path):
    gdown.download(url, model_path, quiet=False)

@st.cache_resource
def load_model():
    return joblib.load(model_path)

model, expected_columns = load_model()

# --- CITY COORDINATES ---
city_coords = {
    'Delhi': [28.6139, 77.2090],
    'Mumbai': [19.0760, 72.8777],
    'Bangalore': [12.9716, 77.5946],
    'Kolkata': [22.5726, 88.3639],
    'Hyderabad': [17.3850, 78.4867],
    'Chennai': [13.0827, 80.2707]
}

map_data = pd.DataFrame([
    {'city': city, 'lat': coords[0], 'lon': coords[1]} for city, coords in city_coords.items()
])

# --- UI STYLING ---
st.set_page_config(page_title="Flight Fare Predictor", layout="centered")
st.title("✈️ Flight Fare Prediction")
st.subheader("🧳 Plan smarter. Pay less.")

# --- INTERACTIVE MAP ---
st.subheader("🌍 Select Source and Destination on Map")

selected_cities = st.multiselect(
    "Select exactly TWO cities (First = Source, Second = Destination)",
    list(city_coords.keys()),
    default=["Delhi", "Mumbai"]
)

if len(selected_cities) != 2:
    st.warning("Please select exactly two cities to continue.")
    st.stop()

source, destination = selected_cities
source_coords = city_coords[source]
dest_coords = city_coords[destination]

# --- ROUTE DATA ---
route_data = pd.DataFrame([
    {"from": source, "to": destination,
     "from_lon": source_coords[1], "from_lat": source_coords[0],
     "to_lon": dest_coords[1], "to_lat": dest_coords[0]}
])

# --- DISPLAY MAP ---
st.pydeck_chart(pdk.Deck(
    initial_view_state=pdk.ViewState(
        latitude=(source_coords[0] + dest_coords[0]) / 2,
        longitude=(source_coords[1] + dest_coords[1]) / 2,
        zoom=5,
        pitch=0,
    ),
    layers=[
        pdk.Layer(
            "ScatterplotLayer",
            data=map_data,
            get_position='[lon, lat]',
            get_fill_color='[255, 140, 0, 160]',
            get_radius=60000,
        ),
        pdk.Layer(
            "TextLayer",
            data=map_data,
            get_position='[lon, lat]',
            get_text='city',
            get_size=16,
            get_color=[255, 255, 255],
            get_alignment_baseline="'bottom'",
        ),
        pdk.Layer(
            "LineLayer",
            data=route_data,
            get_source_position='[from_lon, from_lat]',
            get_target_position='[to_lon, to_lat]',
            get_color=[0, 255, 255],
            get_width=5,
        ),
    ]
))

# --- OTHER USER INPUTS ---
airlines = ['SpiceJet', 'AirAsia', 'Vistara', 'GO_FIRST', 'Indigo', 'Air_India']
departure_times = ['Early_Morning', 'Morning', 'Afternoon', 'Evening', 'Night', 'Late_Night']
arrival_times = ['Early_Morning', 'Morning', 'Afternoon', 'Evening', 'Night', 'Late_Night']
stops = ['zero', 'one', 'two_or_more']
classes = ['Economy', 'Business']

with st.expander("🎛️ Flight Details", expanded=True):
    col1, col2 = st.columns(2)
    with col1:
        airline = st.selectbox("✈️ Airline", airlines)
        stop_count = st.selectbox("🔁 Number of Stops", stops)
        travel_class = st.selectbox("🎟️ Class", classes)
    with col2:
        departure = st.selectbox("🕓 Departure Time", departure_times)
        arrival = st.selectbox("🕗 Arrival Time", arrival_times)
        duration = st.slider("⏱️ Duration (hrs)", 1.0, 15.0, 2.0)
        days_left = st.slider("📅 Days Until Departure", 1, 60, 30)

# --- PREDICTION ---
if st.button("🚀 Predict Flight Fare"):
    with st.spinner("Calculating your fare..."):
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
