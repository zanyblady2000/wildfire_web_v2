import streamlit as st
import joblib
import pandas as pd
import plotly.express as px

# --- Configuration and Asset Loading ---

st.set_page_config(layout="wide", page_title="Simple Weather Prediction App")

@st.cache_resource
def load_assets():
    """Loads the model and scaler only once, using the updated file names."""
    try:
        # --- CORRECTED FILE NAMES HERE ---
        rfc = joblib.load('random_forest_model.joblib') # Corrected model file name
        scaler = joblib.load('scaler (1).pkl')         # Corrected scaler file name
        # -------------------------------
        return rfc, scaler
    except FileNotFoundError:
        st.error("Error: Ensure 'random_forest_model.joblib' and 'scaler (1).pkl' are in the directory.")
        st.stop()
    except Exception as e:
        st.error(f"Error loading assets: {e}")
        st.stop()

rfc, scaler = load_assets()

# --- User Input Function ---

def user_input_features():
    """Collects user inputs via Streamlit sidebar sliders."""
    st.sidebar.header("Input Conditions")
    
    # 1. Predictive Features (Must match the order the scaler was fitted on: temp, humidity, windspeed)
    temp = st.sidebar.slider('1. Temperature (°C)', 0.0, 35.0, 20.0)
    humidity = st.sidebar.slider('2. Humidity (%)', 0.0, 100.0, 50.0)
    windspeed = st.sidebar.slider('3. Windspeed (km/h)', 0.0, 100.0, 15.0)
    
    # 2. Map Features (For visualization only)
    st.sidebar.markdown("---")
    st.sidebar.subheader("Location (Map Visualization)")
    lat = st.sidebar.slider('Latitude', 50.0, 59.0, 55.0)
    long = st.sidebar.slider('Longitude', -124.0, -113.0, -118.0)

    data = {
        'temp': temp, 
        'humidity': humidity, 
        'windspeed': windspeed,
        'lat': lat, 
        'long': long
    }
    return pd.DataFrame(data, index=[0])

# --- Main App Logic ---

st.title("Wildfire Prediction App")
input_df = user_input_features()

# Display input data (optional)
st.subheader('Current Input')
st.dataframe(input_df[['temp', 'humidity', 'windspeed', 'lat', 'long']], use_container_width=True)

# Prediction button and logic
if st.button('Predict Wildfire Outcome', type="primary"):
    
    with st.spinner('Calculating prediction...'):
        # 1. Prepare and scale the 3 required features
        prediction_data = input_df[['temp', 'humidity', 'windspeed']]
        scaled_input = scaler.transform(prediction_data)
        
        # 2. Predict the class (0 or 1) and probabilities
        prediction = rfc.predict(scaled_input)[0]
        proba = rfc.predict_proba(scaled_input)[0]
        
        # 3. Map result for display
        label = {0: 'Low Chance of Fire', 1: 'High Chance of Fire'}
        
        # --- Display Results ---
        st.subheader("Prediction Result")
        st.success(f"The predicted outcome is: **{label}**")
        # --- Map Visualization ---
        
        # Prepare data for map (add prediction label and class)
        map_df = input_df.copy()
        map_df['prediction_label'] = label[prediction]
        
        # Create the map visualization
        fig = px.scatter_mapbox(
            map_df, 
            lat="lat", 
            lon="long", 
            color="prediction_label", 
            color_discrete_map={label[1]: 'red', label[0]: 'green'},
            zoom=8, 
            center={"lat": map_df['lat'].iloc[0], "lon": map_df['long'].iloc[0]},
            height=500,
            mapbox_style="carto-positron", 
            hover_data=['temp', 'humidity', 'windspeed']
        )
        
        st.subheader("Location Visualization")
        st.plotly_chart(fig, use_container_width=True)
