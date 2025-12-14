import streamlit as st
import joblib
import pandas as pd
import plotly.express as px

# --- Load Assets ---
try:
    # --- MODIFIED FILE NAMES HERE ---
    rfc = joblib.load('random_forest_model.joblib')
    scaler = joblib.load('minmax_scalers.joblib') 
    # --------------------------------

except FileNotFoundError:
    # Updated error message to reflect new file names
    st.error("Error: Ensure 'random_forest_model.joblib' and 'minmax_scalers.joblib' are in the same folder.")
    st.stop()

st.title("Weather Prediction App (RFC Model)")
st.sidebar.header("Input Weather Conditions")

# --- User Input Function ---
def user_input_features():
    """Collects user inputs via Streamlit sidebar sliders."""
    temp = st.sidebar.slider('Temperature (°C)', -10.0, 40.0, 20.0)
    humidity = st.sidebar.slider('Humidity (%)', 0.0, 100.0, 50.0)
    windspeed = st.sidebar.slider('Windspeed (km/h)', 0.0, 50.0, 15.0)
    lat = st.sidebar.slider('Latitude', 0.0, 59.0, 50.0)
    long = st.sidebar.slider('Longitude', -180.0, 180.0, -100.0) 

    data = {'temp': temp, 'humidity': humidity, 'windspeed': windspeed,
            'lat': lat, 'long': long}
            
    features_df = pd.DataFrame(data, index=[0]) 
    return features_df

# --- Main App Logic ---

# Define the raw input DataFrame (this runs every time)
raw_input_df = user_input_features() 

st.subheader('User Input Features (Raw)')
st.write(raw_input_df)

# The code INSIDE this 'if' block ONLY runs when the button is clicked:
if st.button('Predict Outcome'):
    
    # 1. Prepare data for model prediction (only the 3 features)
    prediction_data = raw_input_df[['temp', 'humidity', 'windspeed']]
    scaled_input_array = scaler.transform(prediction_data)
    prediction = rfc.predict(scaled_input_array)
    predicted_value = prediction 
    
    st.subheader('Prediction Result')
    st.success(f"The model predicts: {predicted_value}")
    
    # 2. Prepare data for Plotly map visualization
    map_data = raw_input_df.copy()
    map_data['prediction_value'] = predicted_value
    prediction_mapping = {0: 'Low', 1: 'High'} 

    # 3. Create the Plotly figure using the 'map_data' DataFrame:
    fig = px.scatter_mapbox(
        map_data, # Pass the prepared DataFrame here
        lat="lat", 
        lon="long", 
        color="prediction_value", 
        color_discrete_map={'High': 'red', 'Low': 'green'}, 
        zoom=5,             
        height=400,
        mapbox_style="carto-positron", 
        hover_data=['temp', 'humidity', 'windspeed', 'prediction_value'] 
    )
    
    # 4. Display the figure using st.plotly_chart
    st.plotly_chart(fig, use_container_width=True)
