import joblib
import pandas as pd
import numpy as np

# Load the trained model and scalers
model = joblib.load('random_forest_model.joblib')
scalers = joblib.load('minmax_scalers.joblib')

# Extract individual scalers
scaler_temp = scalers['temp_scaler']
spec_scaler_humidity = scalers['humidity_scaler']
spec_scaler_wind_speed = scalers['windspeed_scaler']

def preprocess_input(temp, humidity, windspeed):
    """
    Preprocesses new input data using the loaded MinMaxScaler objects.
    """
    # Scale the input features. To avoid UserWarning, provide inputs as DataFrame with original feature names.
    scaled_temp = scaler_temp.transform(pd.DataFrame([[temp]], columns=['feature_0']))
    scaled_humidity = spec_scaler_humidity.transform(pd.DataFrame([[humidity]], columns=['feature_1']))
    scaled_windspeed = spec_scaler_wind_speed.transform(pd.DataFrame([[windspeed]], columns=['feature_2']))

    # Create a DataFrame for prediction
    preprocessed_data = pd.DataFrame({
        'temp': scaled_temp.flatten(),
        'humidity': scaled_humidity.flatten(),
        'windspeed': scaled_windspeed.flatten()
    })
    return preprocessed_data

def predict_fire_occurrence(temp, humidity, windspeed):
    """
    Predicts fire occurrence based on temperature, humidity, and wind speed.
    """
    # Preprocess the input data
    processed_input = preprocess_input(temp, humidity, windspeed)

    # Make prediction
    prediction = model.predict(processed_input)
    return prediction[0]

if __name__ == '__main__':
    print("--- app.py started ---")
    # Example usage:
    temp1, humidity1, windspeed1 = 15, 60, 20
    prediction1 = predict_fire_occurrence(temp1, humidity1, windspeed1)
    print(f"For Temp={temp1}°C, Humidity={humidity1}%, Wind Speed={windspeed1} km/h: Fire Occurrence = {prediction1}")

    temp2, humidity2, windspeed2 = 32, 25, 45
    prediction2 = predict_fire_occurrence(temp2, humidity2, windspeed2)
    print(f"For Temp={temp2}°C, Humidity={humidity2}%, Wind Speed={windspeed2} km/h: Fire Occurrence = {prediction2}")

    temp3, humidity3, windspeed3 = 28, 35, 25
    prediction3 = predict_fire_occurrence(temp3, humidity3, windspeed3)
    print(f"For Temp={temp3}°C, Humidity={humidity3}%, Wind Speed={windspeed3} km/h: Fire Occurrence = {prediction3}")