import streamlit as st
import joblib
import pandas as pd
import plotly.express as px


rfc = joblib.load('random_forest_model.joblib')
scaler = joblib.load('scaler (1).pkl')         

def Inputs():
  st.sidebar.header("Inputs For Prediction")

  temp = st.sidebar.slider('Temperature (C*)', 15, 35, 20)
  humidity = st.sidebar.slider('Humidity (%)', 0, 100, 35)
  windspeed = st.sidebar.slider('windspeed (Km/h)', 0, 50, 15)

  st.sidebar.subheader("Location Inputs")
  lat = st.sidebar.slider('Latitude', 50, 59, 55)
  long = st.sidebar.slider('Longitude', -124, -113, -118)

  data = {
        'temp': temp, 
        'humidity': humidity, 
        'windspeed': windspeed,
        'lat': lat, 
        'long': long
        }
  return pd.DataFrame(data, index=[0])

st.title("Flame Cast")
input_df = Inputs()

if st.button('Predict Wildfire Within Area'):
  prediction_data = input_df[['temp', 'humidity', 'windspeed']]
  scaled_input = scaler.transform(prediction_data)

  prediction = rfc.predict(scaled_input)

  prediction_label = {0: 'Low Chance of Fire', 1: 'High Chance of Fire'}

  st.subheader("Predicted Result:")
  fire_risk_label = "High" if prediction[0] == 1 else "Low"

  if fire_risk_label == "High":
      st.error(f"Predicted Fire Risk: **{fire_risk_label}**")
  else:
     st.success(f"Predicted Fire Risk: **{fire_risk_label}**")
      
            
     map_df = input_df.copy()
     map_df['prediction_label'] = fire_risk_label[prediction.item()]
        
       
     fig = px.scatter_mapbox(
            map_df, 
            lat="lat",
            lon="long", 
            color="prediction_label", 
            color_discrete_map={fire_risk_label[1]: 'red', fire_risk_label[0]: 'green'},
            zoom=8, 
            center={"lat": map_df['lat'].iloc[0], "lon": map_df['long'].iloc[0]},
            height=500,
            mapbox_style="carto-positron", 
            hover_data=['temp', 'humidity', 'windspeed']
        )
        
     st.subheader("Location Visualization")
     st.plotly_chart(fig, use_container_width=True)
