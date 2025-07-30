import streamlit as st
import pandas as pd
import numpy as np
import pydeck as pdk
import joblib
import gdown
import os

# --- MODEL LOADING ---
# Google Drive file ID for the pre-trained model
file_id = "1KwIruPv9z2U_T86Lpv1KNkmfcUOKyKNj"
url = f"https://drive.google.com/uc?id={file_id}"
model_path = "flight_fare_model_rf.pkl"

# Download the model if it doesn't already exist locally
if not os.path.exists(model_path):
    st.info(f"Downloading machine learning model from Google Drive...")
    try:
        gdown.download(url, model_path, quiet=False)
        st.success("Model downloaded successfully!")
    except Exception as e:
        st.error(f"Error downloading model: {e}. Please ensure you have internet access.")
        st.stop() # Stop the app if model cannot be loaded

# Cache the model loading to prevent re-loading on every rerun
@st.cache_resource
def load_model():
    """Loads the pre-trained Random Forest model and its expected columns."""
    try:
        model_data = joblib.load(model_path)
        # Assuming the loaded object is a tuple (model, expected_columns)
        if isinstance(model_data, tuple) and len(model_data) == 2:
            return model_data
        else:
            # If the loaded object is just the model, assume expected_columns needs to be handled differently
            # or is not provided in the pkl file. For this example, we'll raise an error.
            raise ValueError("Loaded model file does not contain expected_columns.")
    except Exception as e:
        st.error(f"Error loading model: {e}. Please check the model file.")
        st.stop()

model, expected_columns = load_model()

# --- CITY COORDINATES ---
# Dictionary mapping city names to their approximate latitude and longitude
city_coords = {
    'Delhi': [28.6139, 77.2090],
    'Mumbai': [19.0760, 72.8777],
    'Bangalore': [12.9716, 77.5946],
    'Kolkata': [22.5726, 88.3639],
    'Hyderabad': [17.3850, 78.4867],
    'Chennai': [13.0827, 80.2707]
}

# Create a DataFrame for map data, including city names and coordinates
map_data = pd.DataFrame([
    {'city': city, 'lat': coords[0], 'lon': coords[1]} for city, coords in city_coords.items()
])

# --- UI STYLING ---
# Set Streamlit page configuration for better layout and title
st.set_page_config(page_title="Flight Fare Predictor", layout="wide") # Changed layout to 'wide' for better side-by-side
st.title("✈️ Flight Fare Prediction")
st.subheader("🧳 Plan smarter. Pay less.")

# --- CITY SELECTION (MOVED ABOVE COLUMNS FOR ALIGNMENT) ---
st.subheader("🌍 Select Source and Destination")

# Multiselect widget for users to choose source and destination cities
selected_cities = st.multiselect(
    "Select exactly TWO cities (First = Source, Second = Destination)",
    list(city_coords.keys()),
    default=["Delhi", "Mumbai"],
    key="city_select" # Added a key to prevent potential issues with multiple selectboxes
)

# Enforce selection of exactly two cities
if len(selected_cities) != 2:
    st.warning("Please select exactly two cities to continue.")
    st.stop() # Stop execution until valid selection is made

source, destination = selected_cities
source_coords = city_coords[source]
dest_coords = city_coords[destination]

# --- MAIN LAYOUT: TWO COLUMNS ---
# Create two columns: one for the map (left) and one for inputs (right)
col_map, col_inputs = st.columns([0.6, 0.4]) # Adjust ratio as needed

with col_map:
    # --- ROUTE DATA ---
    # DataFrame to hold the source and destination coordinates for the line layer
    route_data = pd.DataFrame([
        {"from": source, "to": destination,
         "from_lon": source_coords[1], "from_lat": source_coords[0],
         "to_lon": dest_coords[1], "to_lat": dest_coords[0]}
    ])

    # --- DISPLAY MAP ---
    # PyDeck chart to visualize the selected cities and the flight route
    st.pydeck_chart(pdk.Deck(
        initial_view_state=pdk.ViewState(
            latitude=(source_coords[0] + dest_coords[0]) / 2, # Center map between cities
            longitude=(source_coords[1] + dest_coords[1]) / 2,
            zoom=5,
            pitch=0,
        ),
        layers=[
            # Layer for city markers (scatterplots)
            pdk.Layer(
                "ScatterplotLayer",
                data=map_data,
                get_position='[lon, lat]',
                get_fill_color='[255, 140, 0, 160]', # Orange color for cities
                get_radius=60000, # Radius in meters
            ),
            # Layer for city names (text labels)
            pdk.Layer(
                "TextLayer",
                data=map_data,
                get_position='[lon, lat]',
                get_text='city',
                get_size=16,
                get_color=[255, 255, 255], # White text
                get_alignment_baseline="'bottom'",
            ),
            # Layer for the flight route line (white line)
            pdk.Layer(
                "LineLayer",
                data=route_data,
                get_source_position='[from_lon, from_lat]',
                get_target_position='[to_lon, to_lat]',
                get_color=[255, 255, 255], # White line
                get_width=8, # Thicker line for visibility
            ),
            # Layer for the airplane icon at the destination
            pdk.Layer(
                "TextLayer",
                data=pd.DataFrame([{'text': '✈️', 'lat': dest_coords[0], 'lon': dest_coords[1]}]),
                get_position='[lon, lat]',
                get_text='text',
                get_size=30, # Adjust size of the emoji
                get_color=[255, 255, 255], # White color for the emoji
                get_pixel_offset=[0, -30], # Offset to place it slightly above the destination point
                get_alignment_baseline="'bottom'",
            ),
        ]
    ))

with col_inputs:
    # --- OTHER USER INPUTS ---
    # Define options for various flight details
    airlines = ['SpiceJet', 'AirAsia', 'Vistara', 'GO_FIRST', 'Indigo', 'Air_India']
    departure_times = ['Early_Morning', 'Morning', 'Afternoon', 'Evening', 'Night', 'Late_Night']
    arrival_times = ['Early_Morning', 'Morning', 'Afternoon', 'Evening', 'Night', 'Late_Night']
    stops = ['zero', 'one', 'two_or_more']
    classes = ['Economy', 'Business']

    with st.expander("🎛️ Flight Details", expanded=True):
        col1, col2 = st.columns(2) # Use two columns for better layout within the expander
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
    # Button to trigger fare prediction
    if st.button("🚀 Predict Flight Fare"):
        with st.spinner("Calculating your fare..."):
            # Create a DataFrame from user inputs
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

            # One-hot encode categorical features
            input_df = pd.get_dummies(input_df)
            # Reindex the DataFrame to match the columns the model expects, filling missing with 0
            input_df = input_df.reindex(columns=expected_columns, fill_value=0)

            # Make prediction using the loaded model
            predicted_fare = model.predict(input_df)[0]
            st.success(f"✅ Estimated Fare: ₹{predicted_fare:,.2f}")

# --- CUSTOM CSS FOR BACKGROUND IMAGE ---
# This CSS block targets the main Streamlit container and sets the background image.
# It uses 'cover' to ensure the image covers the entire background, 'center' to center it,
# and 'no-repeat' to prevent tiling.
# Then apply the CSS styling as before
st.markdown('<div class="overlay"></div>', unsafe_allow_html=True)

st.markdown(
     f"""
     <style>
     .stApp {{
         background-image: url("https://images.unsplash.com/photo-1569154941061-e231b4725ef1?q=80&w=2670&auto=format&fit=crop&ixlib=rb-4.1.0&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D");
         background-size: cover;
         background-position: center;
         background-repeat: no-repeat;
         background-attachment: fixed;
         position: relative;
         overflow: hidden;
     }}
     .overlay {{
         position: fixed;
         top: 0;
         left: 0;
         width: 100%;
         height: 100%;
         background-color: rgba(0, 0, 0, 0.4);
         z-index: -1;
     }}
     </style>
     """,
     unsafe_allow_html=True
)

