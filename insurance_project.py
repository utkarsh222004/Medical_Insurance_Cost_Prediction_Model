# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import streamlit as st

# Load the data
data = pd.read_csv('insurance.csv')

# Feature engineering and preprocessing
data['smoker'] = data['smoker'].map({'yes': 1, 'no': 0})
data = pd.get_dummies(data, columns=['sex', 'region'], drop_first=True)

# Ensure 'northeast' is included in the one-hot encoding
if 'region_northeast' not in data.columns:
    data['region_northeast'] = 0

# Split the data into features and target
X = data[['age', 'sex_male', 'bmi', 'children', 'smoker', 'region_northeast', 'region_northwest', 'region_southeast', 'region_southwest']]
y = data['charges']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Linear Regression
linear_reg = LinearRegression()
linear_reg.fit(X_train_scaled, y_train)

# Random Forest
random_forest = RandomForestRegressor()
random_forest.fit(X_train_scaled, y_train)

# Neural Network
model = Sequential()
model.add(Dense(64, input_dim=X_train_scaled.shape[1], activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='linear'))
model.compile(optimizer='adam', loss='mean_squared_error')

model.fit(X_train_scaled, y_train, epochs=50, batch_size=32, verbose=0)

# Streamlit App
st.title('Medical Insurance Cost Prediction')

# Sidebar for user input
st.header('Enter Patient Information')

# Example input data
default_age = 33
default_sex = 'male'
default_bmi = 22.705
default_children = 0
default_smoker = 'no'
default_region = 'northwest'

age = st.number_input('Age', min_value=18, max_value=64, value=default_age)
sex = st.radio('Gender', ('Male', 'Female'), index=0 if default_sex.lower() == 'male' else 1)
bmi = st.number_input('BMI', min_value=15.0, max_value=50.0, value=default_bmi)
children = st.slider('Number of Children', min_value=0, max_value=5, value=default_children)
smoker = st.radio('Smoker', ('Yes', 'No'), index=1 if default_smoker.lower() == 'no' else 0)
region = st.selectbox('Region', ('Northeast', 'Northwest', 'Southeast', 'Southwest'), index=1 if default_region.lower() == 'northwest' else 0)

# Map categorical variables
sex = 1 if sex == 'Male' else 0
smoker = 1 if smoker == 'Yes' else 0
region_mapping = {'Northeast': 0, 'Northwest': 1, 'Southeast': 2, 'Southwest': 3}
region = region_mapping[region]

# Create input DataFrame
input_data = pd.DataFrame([[age, sex, bmi, children, smoker, *([0]*region + [1] + [0]*(3-region))]],
                          columns=['age', 'sex_male', 'bmi', 'children', 'smoker', 'region_northeast', 'region_northwest', 'region_southeast', 'region_southwest'])

# Standardize input data
input_data_scaled = scaler.transform(input_data)

# Predictions
linear_reg_prediction = linear_reg.predict(input_data_scaled)[0]
random_forest_prediction = random_forest.predict(input_data_scaled)[0]
nn_prediction = model.predict(input_data_scaled).flatten()[0]

# Display predictions
st.write('### Predicted Medical Insurance Cost By:')
st.write(f'Linear Regression is : ${linear_reg_prediction:.2f}')
st.write(f'Random Forest is : ${random_forest_prediction:.2f}')
st.write(f'Neural Network is : ${nn_prediction:.2f}')



# Apply red background
style = """
    <style>
        div[data-baseweb="card"] {
            background-color: red;
        }
    </style>
"""
st.markdown(style, unsafe_allow_html=True)

