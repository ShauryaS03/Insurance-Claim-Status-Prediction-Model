import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE
import numpy as np

st.write("# Insurance Claim Status")
st.write('---')

# Load data
data = pd.read_csv(r"D:\Insurance Claim Status Project\InsuranceClaims.csv")
data = data[data['segment'] != 'Utility']

categorical_col = data.select_dtypes(include=['object']).columns
label_encoders = {}
for col in categorical_col:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    label_encoders[col] = le

X = data.drop(columns=['claim_status'])
y = data['claim_status']

# Balance and scale data
# smote = SMOTE(random_state=42)
# X_balanced, y_balanced = smote.fit_resample(X, y)

# numerical_columns = X.select_dtypes(include=['float64', 'int64']).columns
# scaler = StandardScaler()
# X_balanced[numerical_columns] = scaler.fit_transform(X_balanced[numerical_columns])


Region_code = ['C8', 'C2', 'C10', 'C13', 'C7', 'C5', 'C3', 'C19', 'C9', 'C15', 'C6', 'C11', 'C1', 'C14', 'C17', 'C12', 'C4', 'C21', 'C16', 'C18', 'C22', 'C20']
Segment = ['C2', 'C1', 'A', 'B2', 'B1']
Model = ['M4', 'M9', 'M1', 'M5', 'M7', 'M6', 'M8', 'M3', 'M2', 'M11']
Fuel_type = ['Diesel', 'CNG', 'Petrol']
Is_adjustable_steering = ['Yes', 'No']
Is_tpms = ['Yes', 'No']
Is_parking_sensors = ['Yes', 'No']
Is_parking_camera = ['Yes', 'No']
Rear_brakes_type = ['Disc', 'Drum']
Transmission_type = ['Automatic', 'Manual']
Is_brake_assist = ['Yes', 'No']
Is_central_locking = ['Yes', 'No']
Is_speed_alert = ['Yes', 'No']


# Sidebar for user inputs
st.sidebar.header('Select Input Parameters')

def user_input_features():
    region_code = st.selectbox('Region Code', Region_code, key="region_code")
    segment = st.selectbox('Segment', Segment, key="segment")
    model = st.selectbox('Model', Model, key="model")
    fuel_type = st.selectbox('Fuel Type', Fuel_type, key="fuel_type")
    adjustable_steering = st.selectbox('Adjustable Steering', Is_adjustable_steering, key="adjustable_steering")
    tpms = st.selectbox('TPMS', Is_tpms, key="tpms")
    parking_sensors = st.selectbox('Parking Sensors', Is_parking_sensors, key="parking_sensors")
    parking_camera = st.selectbox('Parking Camera', Is_parking_camera, key="parking_camera")
    rear_brakes_type = st.selectbox('Rear Brakes Type', Rear_brakes_type, key="rear_brakes_type")
    transmission_type = st.selectbox('Transmission Type', Transmission_type, key="transmission_type")
    brake_assist = st.selectbox('Brake Assist', Is_brake_assist, key="brake_assist")
    central_locking = st.selectbox('Central Locking', Is_central_locking, key="central_locking")
    speed_alert = st.selectbox('Speed Alert', Is_speed_alert, key="speed_alert")
    subscription_length = st.sidebar.slider('Subscription Length', 0.0, 14.0, 6.3, key="subscription_length")
    vehicle_age = st.sidebar.slider('Vehicle Age', 0.0, 20.0, 1.52, key="vehicle_age")
    max_torque = st.sidebar.slider('Max Torque', 60.0, 250.0, 139.11, key="max_torque")
    max_power = st.sidebar.slider('Max Power', 40.36, 118.36, 81.55, key="max_power")
    airbags = st.sidebar.slider('Airbags', 2, 6, 3, key="airbags")
    displacement = st.sidebar.slider('Displacement', 796, 1498, 1185, key="displacement")
    cylinder = st.sidebar.slider('Cylinder', 3, 4, 3, key="cylinder")
    turning_radius = st.sidebar.slider('Turning Radius', 4.6, 5.2, 4.88, key="turning_radius")
    gross_weight = st.sidebar.slider('Gross Weight', 1051, 1720, 1387, key="gross_weight")
    ncap_rating = st.sidebar.slider('NCAP Rating', 0, 5, 2, key="ncap_rating")

    # Combine inputs into a DataFrame
    df = pd.DataFrame({
        "subscription_length": [subscription_length],
        "vehicle_age": [vehicle_age],
        "region_code": [region_code],
        "segment": [segment],
        "model": [model],
        "fuel_type": [fuel_type],
        "max_torque": [max_torque],
        "max_power": [max_power],
        "airbags": [airbags],
        "is_adjustable_steering": [adjustable_steering],
        "is_tpms": [tpms],
        "is_parking_sensors": [parking_sensors],
        "is_parking_camera": [parking_camera],
        "rear_brakes_type": [rear_brakes_type],
        "displacement": [displacement],
        "cylinder": [cylinder],
        "transmission_type": [transmission_type],
        "turning_radius": [turning_radius],
        "gross_weight": [gross_weight],
        "is_brake_assist": [brake_assist],
        "is_central_locking": [central_locking],
        "is_speed_alert": [speed_alert],
        "ncap_rating": [ncap_rating]
    })

    # Preprocess the data
    categorical_columns = df.select_dtypes(include=['object']).columns
    for col in categorical_columns:
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

    # # Scale numerical columns
    # numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns
    # df[numerical_columns] = scaler.fit_transform(df[numerical_columns])
    
    # x_balanced = smote.fit_resample(df)

    return df

df1 = user_input_features()

# Train and predict
model1 = LogisticRegression()
model1.fit(X,y)

pred = model1.predict(df1)
st.write('---')

st.header('Specified Input Parameters.')
st.write(df1)
st.write('---')

probabilities = np.round(model1.predict_proba(df1),4) 

st.header('Prediction')
st.write(f"Predicted Claim Status: **{pred[0]}**")
st.write(f"Predicted Probabilities: **{probabilities}**")


