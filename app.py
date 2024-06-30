

import streamlit as st
import pandas as pd
import pickle
import datetime
import os

# Define paths based on the new directory structure
model_path = os.path.join('models', 'taxi_fare_model.pkl')
columns_path = os.path.join('models', 'model_columns.pkl')
mean_values_path = os.path.join('utils', 'mean_values.csv')

# Load the trained model and column names
with open(model_path, 'rb') as file:
    model = pickle.load(file)

with open(columns_path, 'rb') as file:
    model_columns = pickle.load(file)

# Load the mean values DataFrame
# Ensure this CSV file exists and has the necessary data
mean_values = pd.read_csv(mean_values_path)

#######

# Define a function to preprocess user input
def preprocess_input(data):
    data['VendorID'] = data['VendorID'].astype(str)
    data['rush_hour'] = ((data['tpep_pickup_datetime'].dt.hour >= 7) & (data['tpep_pickup_datetime'].dt.hour <= 9) & 
                         (data['tpep_pickup_datetime'].dt.dayofweek < 5)) | ((data['tpep_pickup_datetime'].dt.hour >= 16) & 
                         (data['tpep_pickup_datetime'].dt.hour <= 18) & (data['tpep_pickup_datetime'].dt.dayofweek < 5)).astype(int)
    data = pd.get_dummies(data, drop_first=True)
    data = data.reindex(columns=model_columns, fill_value=0)
    return data

# Streamlit app
st.title('NYC Taxi Fare Predictor')

# User inputs
pickup_datetime = st.date_input("Pickup Date", datetime.date(2023, 1, 1))
pickup_time = st.time_input("Pickup Time", datetime.time(8, 0))
vendor_id = st.selectbox("Vendor ID", ['1', '2'])
rate_code = st.selectbox("Rate Code ID", [1, 2, 3, 4, 5, 6])
passenger_count = st.number_input("Passenger Count", min_value=1, max_value=10, value=1)
pu_location_id = st.number_input("Pickup Location ID", min_value=1, max_value=265, value=1)
do_location_id = st.number_input("Dropoff Location ID", min_value=1, max_value=265, value=1)

# Calculate mean distance and mean duration (using the current data as a proxy)
pickup_dropoff = str(pu_location_id) + ' ' + str(do_location_id)
# Fetch mean distance and duration
try:
  # Fetch mean distance and duration
  mean_distance = mean_values.loc[mean_values['pickup_dropoff'] == pickup_dropoff, 'mean_distance'].values[0]
  mean_duration = mean_values.loc[mean_values['pickup_dropoff'] == pickup_dropoff, 'mean_duration'].values[0]
except KeyError:
  # Handle case where pickup_dropoff combination is not found
  st.error("Invalid Pickup or Dropoff combination. Please try again.")
  mean_distance = 0  # Set placeholder values for prediction (can be adjusted based on your model's needs)
  mean_duration = 0


# Prepare the input data for prediction
input_data = pd.DataFrame({
    'VendorID': [vendor_id],
    'tpep_pickup_datetime': [pd.to_datetime(str(pickup_datetime) + ' ' + str(pickup_time))],
    'passenger_count': [passenger_count],
    'RatecodeID': [rate_code],
    'PULocationID': [pu_location_id],
    'DOLocationID': [do_location_id],
    'mean_distance': [mean_distance], # Use retrieved values or placeholders
    'mean_duration': [mean_duration]  # Use retrieved values or placeholders
})

# Preprocess the input data
input_data_preprocessed = preprocess_input(input_data)

# Make predictions
if st.button('Predict Fare'):
    fare_prediction = model.predict(input_data_preprocessed)
    st.write(f"Predicted Fare Amount: ${fare_prediction[0]:.2f}")


#streamlit run app.py


