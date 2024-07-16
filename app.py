import streamlit as st
import pandas as pd
import pickle
import datetime
import os


@st.cache_resource
def load_data():
  # Load the trained model and column names
  model_path = os.path.join('models', 'taxi_fare_model.pkl')
  with open(model_path, 'rb') as file:
    model = pickle.load(file)

  model_columns_path = os.path.join('models', 'model_columns.pkl')
  with open(model_columns_path, 'rb') as file:
    model_columns = pickle.load(file)

  # Load the mean values DataFrame
  mean_values_path = os.path.join('data', 'processed', 'mean_values.csv')
  mean_values = pd.read_csv(mean_values_path)

  return model, model_columns, mean_values


# Define a function to preprocess user input
def preprocess_input(data, model_columns):
  data['pickup_hour'] = data['tpep_pickup_datetime'].dt.hour  # Hour of the day
  data['pickup_day_of_week'] = data['tpep_pickup_datetime'].dt.dayofweek  # Day of the week (0=Monday, 6=Sunday)
  data['pickup_month'] = data['tpep_pickup_datetime'].dt.month  # Month (1=January, 12=December)
  data = pd.get_dummies(data, drop_first=True)  # One-hot encode categorical features
  data = data.reindex(columns=model_columns, fill_value=0)  # Impute missing values with 0
  return data


def time_of_day(data):
  if 6 <= data < 12:
    return 'Morning'
  elif 12 <= data < 16:
    return 'Afternoon'
  elif 16 <= data < 22:
    return 'Evening'
  else:
    return 'Late night'


# Streamlit app
st.title('NYC Taxi Fare Predictor')

# Load data
model, model_columns, mean_values = load_data()

# Get current date and time
now = datetime.datetime.now()   



# User inputs
pickup_date = st.date_input("Pickup Date", now.date())
pickup_time_str = st.text_input("Pickup Time (HH:MM)", now.strftime("%H:%M"))

# Combine date and time
try:
    # Parse the time input
    pickup_time = datetime.datetime.strptime(pickup_time_str, "%H:%M").time()
    
    # Validate hour and minute ranges
    if not (0 <= pickup_time.hour <= 23 and 0 <= pickup_time.minute <= 59):
        st.error("Invalid time format. Hours must be between 0 and 23, minutes between 0 and 59.")
        pickup_time = None  # Reset pickup_time to None for invalid input

    # Combine date and time if valid
    if pickup_time:
        pickup_datetime = pd.to_datetime(datetime.datetime.combine(pickup_date, pickup_time))
        st.write("Pickup DateTime:", pickup_datetime)
except ValueError:
    st.error("Invalid time format. Please use HH:MM format.")
    pickup_time = None

########

# Create a dictionary to map descriptive vendor names to their numerical IDs.
vendor_id_options = {
    "1= Creative Mobile Technologies, LLC": 1,
    "2= VeriFone Inc.": 2
}

vendor_id_selected = st.selectbox("Vendor ID", list(vendor_id_options.keys()))

# Extract the corresponding numerical vendor ID based on the user's selection.
vendor_id = vendor_id_options[vendor_id_selected]

# Create a selectbox to allow users to choose a rate code ID.
rate_code_options = [
    "1 = Standard rate",
    "2 = JFK",
    "3 = Newark",
    "4 = Nassau or Westchester",
    "5 = Negotiated fare",
    "6 = Group ride"
]
rate_code_selected = st.selectbox("Rate Code ID", rate_code_options)  # Display rate code options

# Extract the numerical rate code ID from the selected option.
rate_code = int(rate_code_selected.split("=")[0])  # Convert selected option to integer


passenger_count = st.number_input("Passenger Count", min_value=1, max_value=6, step=1)


# Get unique pickup locations
pickup_locations = mean_values['PULocationID'].unique()

# Create the pickup location selectbox
pu_location_id = st.selectbox('Select Pickup Location', pickup_locations)

# Filter mean_values based on selected pickup location
filtered_data = mean_values[mean_values['PULocationID'] == pu_location_id]
dropoff_locations = filtered_data['DOLocationID'].unique()

# Create the drop-off location selectbox
do_location_id = st.selectbox('Select Drop-off Location', dropoff_locations)

# Fetch mean distance and duration
try:
  mean_distance = mean_values[(mean_values['PULocationID'] == pu_location_id) & (mean_values['DOLocationID'] == do_location_id)]['mean_distance'].values[0]
  mean_duration = mean_values[(mean_values['PULocationID'] == pu_location_id) & (mean_values['DOLocationID'] == do_location_id)]['mean_duration'].values[0]
except IndexError:
  st.error("Invalid Pickup or Dropoff combination. Please try again.")
  # Exit the code execution here to prevent prediction with potentially invalid data

# Prepare the input data for prediction
input_data = pd.DataFrame({
  'VendorID': [vendor_id],
  'tpep_pickup_datetime': [pickup_datetime],
  'passenger_count': [passenger_count],
  'RatecodeID': [rate_code],
  'PULocationID': [pu_location_id],
  'DOLocationID': [do_location_id],
  'mean_distance': [mean_distance],
  'mean_duration': [mean_duration]
  })

input_data['pickup_hour'] = input_data['tpep_pickup_datetime'].dt.hour
input_data['pickup_day_of_week'] = input_data['tpep_pickup_datetime'].dt.dayofweek
input_data['pickup_month'] = input_data['tpep_pickup_datetime'].dt.month
input_data['pickup_timeofday'] = input_data['pickup_hour'].apply(time_of_day)

# Check for missing inputs
if None in [pickup_datetime, pickup_time, vendor_id, rate_code, passenger_count, pu_location_id, do_location_id]:
    st.error("Please fill in all required fields before predicting the fare.")
else:
    # Preprocess the input data
    input_data_preprocessed = preprocess_input(input_data, model_columns)

    # Make predictions
    if st.button('Predict Fare'):
        fare_prediction = model.predict(input_data_preprocessed)
        st.write(f"Predicted Fare Amount: ${fare_prediction[0]:.2f}")

#streamlit run app.py