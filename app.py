import streamlit as st
import joblib
import pandas as pd
import plotly.express as px


st.set_page_config(layout="wide", page_title="Wildfire Prediction Web-app")

      
 rfc = joblib.load('random_forest_model.joblib')
 scaler = joblib.load('scaler (1).pkl')         

def user_input_features():
    st.sidebar.header("Input Conditions")
    
    temp = st.sidebar.slider('1. Temperature (°C)', 0.0, 35.0, 20.0)
    humidity = st.sidebar.slider('2. Humidity (%)', 0.0, 100.0, 50.0)
    windspeed = st.sidebar.slider('3. Windspeed (km/h)', 0.0, 100.0, 15.0)
    

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

st.title("Wildfire Prediction App")
input_df = user_input_features()

st.subheader('Current Input')
st.dataframe(input_df[['temp', 'humidity', 'windspeed', 'lat', 'long']], use_container_width=True)

if st.button('Predict Wildfire Outcome'):
    
    with st.spinner('Calculating prediction...'):
        
        prediction_data = input_df[['temp', 'humidity', 'windspeed']]
        scaled_input = scaler.transform(prediction_data)
        
    
        prediction = rfc.predict(scaled_input)[0]
        proba = rfc.predict_proba(scaled_input)[0]
        
      
        label = {0: 'Low Chance of Fire', 1: 'High Chance of Fire'}
        
 
        st.subheader("Prediction Result")
        fire_risk_label = "High" if prediction == 1 else "Low"

        if fire_risk_label == "High":
            st.error(f"Predicted Fire Risk: **{fire_risk_label}**")
        else:
            st.success(f"Predicted Fire Risk: **{fire_risk_label}**")
        

        

        map_df = input_df.copy()
        map_df['prediction_label'] = label[prediction]
        
       
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
