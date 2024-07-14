import streamlit as st
import pandas as pd
import pickle
import datetime
import os

@st.cache_resource
def load_data():
    # Load the trained model and column names
    with open(os.path.join('models', 'taxi_fare_model.pkl'), 'rb') as file:
        model = pickle.load(file)

    with open(os.path.join('models', 'model_columns.pkl'), 'rb') as file:
        model_columns = pickle.load(file)

    # Load the mean values DataFrame
    mean_values = pd.read_csv(os.path.join('data', 'mean_values.csv'))

    # Load pickup and dropoff ID data
    pickup_dropoff_id = pd.read_csv(os.path.join('data', 'pickup_dropoff_id.csv'))

    return model, model_columns, mean_values, pickup_dropoff_id

# Define a function to preprocess user input
def preprocess_input(data, model_columns):
    data['VendorID'] = data['VendorID'].astype(str)
    data['rush_hour'] = ((data['tpep_pickup_datetime'].dt.hour >= 7) & (data['tpep_pickup_datetime'].dt.hour <= 9) & 
                         (data['tpep_pickup_datetime'].dt.dayofweek < 5)) | ((data['tpep_pickup_datetime'].dt.hour >= 16) & 
                         (data['tpep_pickup_datetime'].dt.hour <= 18) & (data['tpep_pickup_datetime'].dt.dayofweek < 5)).astype(int)
    data = pd.get_dummies(data, drop_first=True)
    data = data.reindex(columns=model_columns, fill_value=0)
    return data

# Streamlit app
st.title('NYC Taxi Fare Predictor')

# Load data
model, model_columns, mean_values, pickup_dropoff_id = load_data()

# User inputs
pickup_datetime = st.date_input("Pickup Date", datetime.date(2023, 1, 1))
pickup_time = st.time_input("Pickup Time", datetime.time(8, 0))
vendor_id = st.selectbox("Vendor ID", ['1', '2'])
rate_code = st.selectbox("Rate Code ID", [1, 2, 3, 4, 5, 6])
passenger_count = st.selectbox("Passenger Count", [1, 2, 3, 4, 5, 6])

# Get unique pickup locations
pickup_locations = pickup_dropoff_id['PULocationID'].unique()

# Create the pickup location selectbox
pu_location_id = st.selectbox('Select Pickup Location', pickup_locations)

# Filter the data based on selected pickup location
filtered_data = pickup_dropoff_id[pickup_dropoff_id['PULocationID'] == pu_location_id]
dropoff_locations = filtered_data['DOLocationID'].unique()

# Create the drop-off location selectbox
do_location_id = st.selectbox('Select Drop-off Location', dropoff_locations)

# Calculate mean distance and mean duration (using the current data as a proxy)
pickup_dropoff = str(pu_location_id) + ' ' + str(do_location_id)

# Fetch mean distance and duration
if pickup_dropoff in mean_values['pickup_dropoff'].values:
    mean_distance = mean_values.loc[mean_values['pickup_dropoff'] == pickup_dropoff, 'mean_distance'].values[0]
    mean_duration = mean_values.loc[mean_values['pickup_dropoff'] == pickup_dropoff, 'mean_duration'].values[0]
else:
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
    'mean_distance': [mean_distance],
    'mean_duration': [mean_duration]
})

# Preprocess the input data
input_data_preprocessed = preprocess_input(input_data, model_columns)

# Make predictions
if st.button('Predict Fare'):
    fare_prediction = model.predict(input_data_preprocessed)
    st.write(f"Predicted Fare Amount: ${fare_prediction[0]:.2f}")

#streamlit run app.py
