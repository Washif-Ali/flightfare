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
    'Delhi': [77.1025, 28.7041],
    'Mumbai': [72.8777, 19.0760],
    'Bangalore': [77.5946, 12.9716],
    'Kolkata': [88.3639, 22.5726],
    'Hyderabad': [78.4867, 17.3850],
    'Chennai': [80.2707, 13.0827]
}

src_coords = city_coords[source]
dst_coords = city_coords[destination]

# --- UI STYLING ---
st.set_page_config(page_title="Flight Fare Predictor", layout="centered")
st.title("✈️ Flight Fare Prediction")
st.subheader("🧳 Plan smarter. Pay less.")

# --- INTERACTIVE MAP ---
selected_cities = st.multiselect(
    "Select exactly TWO cities (First = Source, Second = Destination)",
    list(city_coords.keys()),
    default=["Delhi", "Mumbai"]
)

if len(selected_cities) != 2:
    st.warning("Please select exactly two cities.")
else:
    source, destination = selected_cities
# Data for line
line_data = pd.DataFrame([{
    "from_lon": src_coords[0],
    "from_lat": src_coords[1],
    "to_lon": dst_coords[0],
    "to_lat": dst_coords[1],
}])

# Data for airplane icon
icon_data = pd.DataFrame([{
    "lat": (src_coords[1] + dst_coords[1]) / 2,
    "lon": (src_coords[0] + dst_coords[0]) / 2,
    "icon_data": {
        "url": "https://upload.wikimedia.org/wikipedia/commons/e/e9/Black_airplane_silhouette.png",
        "width": 128,
        "height": 128,
        "anchorY": 128,
    }
}])

# Define layers
line_layer = pdk.Layer(
    "ArcLayer",
    data=line_data,
    get_source_position=["from_lon", "from_lat"],
    get_target_position=["to_lon", "to_lat"],
    get_width=2,
    get_source_color=[255, 255, 255],
    get_target_color=[255, 255, 255],
    stroke_width=3,
    width_min_pixels=2,
    width_max_pixels=3,
    get_tilt=10,
    pickable=True,
)

icon_layer = pdk.Layer(
    type="IconLayer",
    data=icon_data,
    get_icon="icon_data",
    get_size=4,
    size_scale=10,
    get_position=["lon", "lat"],
    pickable=False,
)

# Render map
st.pydeck_chart(pdk.Deck(
    map_style="mapbox://styles/mapbox/dark-v10",
    initial_view_state=pdk.ViewState(
        latitude=(src_coords[1] + dst_coords[1]) / 2,
        longitude=(src_coords[0] + dst_coords[0]) / 2,
        zoom=4,
        pitch=40,
    ),
    layers=[line_layer, icon_layer],
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
