# NEW YORK CITY TAXI FARE PREDICTION PROJECT REPORT
## Introduction
Since 1971, The New York City Taxi and Limousine Commission (TLC) has been regulating and overseeing the licensing of New York City's taxi cabs, for-hire vehicles, commuter vans, and paratransit vehicles. TLC is now looking to enhance their riders' experience and trust in the service by developing an app that estimates taxi fares in advance of their ride. This project aims to create a machine learning model that predicts taxi fares using historical taxi trip data.

## Project Overview
This project involves data preprocessing, exploratory data analysis (EDA), feature engineering, model training, evaluation, and deployment. The dataset consists of taxi trip records collected over a year and includes key features like trip distance, pickup and drop-off locations, fare amount, and more.

## Data Preprocessing
Steps Taken
1.	Handling Missing Values: There were no missing values in the dataset.
2.	Renaming Columns: The column 'Unnamed: 0' was renamed to 'id'.
3.	Data Type Conversion: Date-time columns were converted to appropriate date and time data types.
4.	Encoding Categorical Variables: The store_and_fwd_flag column was encoded.
5.	Feature Creation: New features such as duration, mean_distance, mean_duration, pickup_hour, pickup_day, pickup_month, pickup_day_of_week, and rush_hour were created to enhance the dataset.
 
![image](https://github.com/mandyiv/NYC_TAXI_FARE_PREDICTION/assets/88114875/346864ea-e875-448a-8dc5-cdf84df217a7)

**Data Info before and after data preprocessing**

![image](https://github.com/mandyiv/NYC_TAXI_FARE_PREDICTION/assets/88114875/68c9a403-c854-4a88-a89b-ed3cd749c882)

![image](https://github.com/mandyiv/NYC_TAXI_FARE_PREDICTION/assets/88114875/a59898fb-0853-4d0f-b094-c5584736142a) 
  

## Exploratory Data Analysis (EDA)
**Descriptive Statistics**
Generated summary statistics for all features to understand their distributions and identify potential outliers.
 ![image](https://github.com/mandyiv/NYC_TAXI_FARE_PREDICTION/assets/88114875/09997d35-1afb-455d-9b66-18b19074a2ea)

➔	**Key Insights from Descriptive Statistics**
Passenger Count
-	It has a mean of 1.64, indicating most rides have 1 or 2 passengers, which is typical for taxi rides.
-	it has a maximum value of 6, reflecting the seating capacity of most standard taxis.
Trip Distance
-	it has a Mean of 2.91 miles, with a wide range (0 to 33.96 miles).
-	The average trip distance is relatively short, indicating that most taxi rides are within a few miles.
-	The maximum distance seems to be an outlier, especially considering the 75th percentile is only 3.06 miles.
RatecodeID
-	it has a mean of 1.04, indicating that most trips are likely to be standard rate (RatecodeID 1).
-	The maximum value is 99, as valid RatecodeIDs range from 1 to 6 according to the data dictionary. 
Store and Forward Flag
-	It has a mean of 0.004. The store-and-forward flag is rarely set to 'Y', indicating that nearly all trips are recorded in real-time ('N').
Payment Type
-	It has a mean of 1.32, induicating that most payments are made via credit card (payment type 1).
-	The data includes all four payment types, reflecting diverse payment methods.
Fare Amount
-	The average fare amount is $13.03, but there is significant variability.
-	The presence of negative values in the range ( -$120 to $999.99) suggests data errors. High values may also include outliers.


## Data Visualization
Plotted distributions and boxplots to visualize data spread and detect outliers. Created correlation heatmaps to understand relationships between feature 
 
 ![image](https://github.com/mandyiv/NYC_TAXI_FARE_PREDICTION/assets/88114875/69751fff-3fbc-4fba-a8d9-016be8e2aada)
 ![image](https://github.com/mandyiv/NYC_TAXI_FARE_PREDICTION/assets/88114875/c04070a7-addc-4e81-9b9a-5bce8c3ef880)
![image](https://github.com/mandyiv/NYC_TAXI_FARE_PREDICTION/assets/88114875/122ed4c6-84bc-4fb1-aae8-7028eb2d04d2)


**Handling Outliers**
The 3 variables shown above have a right-skewed distribution and contain extreme outliers
●	Trip Distance: Trips with zero distance were considered valid as they might reflect immediate cancellations. These instances were insignificant in number (148 out of ~23,000 rides) and therefore left unchanged.
●	Fare Amount: Negative fare values were deleted, while zero fares were retained. Extremely high fares were capped at $62.50 based on an adjusted interquartile range (IQR) method.
●	Trip Duration: Negative durations were deleted. Extremely high durations were capped similarly using an adjusted IQR method.

 ![image](https://github.com/mandyiv/NYC_TAXI_FARE_PREDICTION/assets/88114875/b6130288-afa5-4bc5-bdb6-3f3681fcaa35)

## Feature Engineering
New Features Created
●	Mean Distance and Mean Duration: These features captured average distances and durations for trips with the same pickup and drop-off points.
●	Rush Hour Indicator: A binary column indicating whether the trip occurred during rush hour (7:00 am - 9:00 am and 4:00 pm - 6:00 pm on weekdays).
●	Correlation Analysis: Identified that mean_duration and mean_distance had strong correlations with the target variable (fare amount).

## Model Training
**Feature Selection**
Selected relevant features for model training based on domain knowledge and correlation analysis.
![image](https://github.com/mandyiv/NYC_TAXI_FARE_PREDICTION/assets/88114875/62d6432a-dad4-4bdf-b32c-af8bfe548023)

**Train-Test Split**
Split the data into training and testing sets to evaluate model performance.
 ![image](https://github.com/mandyiv/NYC_TAXI_FARE_PREDICTION/assets/88114875/89489520-a97e-4e1c-95fc-4fd70150b362)


**Model Selection and Training**
●	Baseline Model: Linear Regression.
●	Advanced Model: XGBoost, chosen for its performance and handling of complex data patterns.

**Model Evaluation**
Performance Metrics
●**	XGBoost Model: **
○	Training RMSE: 2.449
○	Testing RMSE: 3.053
○	Training R-squared: 0.940
○	Testing R-squared: 0.909 

**Residuals Analysis**
  
![image](https://github.com/mandyiv/NYC_TAXI_FARE_PREDICTION/assets/88114875/d9ac7e56-d406-43be-b65a-d538d97a2af9)
![image](https://github.com/mandyiv/NYC_TAXI_FARE_PREDICTION/assets/88114875/88fad5f6-fc43-4efc-ad5d-be3df7b17cf3)

●	Scatterplot of Actual vs. Predicted Fares: Most points were close to the line, indicating good predictions. Some deviations were noted, especially for higher fare amounts.
●	Residuals Spread: Residuals were centered around zero, suggesting unbiased predictions. However, variance increased with higher fare amounts, indicating less accuracy for higher fares.

## Next Steps
●	Explore more advanced modeling techniques and additional features to improve prediction accuracy for higher fares.
●	Address the issue of increasing residual variance with more sophisticated models or feature transformations.

## Conclusion
This project successfully developed a machine learning model to predict taxi fares for New York City, achieving high accuracy and providing a reliable tool for TLC's riders. Future improvements can further enhance the model's performance, particularly for higher fare predictions.
