import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load the label encoder, scaler, and model
label_encoder = joblib.load('label_encoder.pkl')
scaler = joblib.load('scaler.pkl')
model = joblib.load('model.pkl')

# Define the Streamlit app
st.title('Conflict Prediction App')

# Get user inputs for the features
actor1 = st.text_input('Actor 1')
actor2 = st.text_input('Actor 2')
actor_dyad_id = st.text_input('Actor Dyad ID')
admin1 = st.text_input('Admin 1')
admin2 = st.text_input('Admin 2')
admin3 = st.text_input('Admin 3')
country = st.text_input('Country')
event_type = st.text_input('Event Type')
geo_precision = st.number_input('Geo Precision')
gwno = st.number_input('GWNO')
inter1 = st.number_input('Inter 1')
inter2 = st.number_input('Inter 2')
interaction = st.number_input('Actor 1')
latitude = st.text_input('Latitude')
location = st.text_input('Location')
longitude = st.text_input('Longitude')
time_precision = st.number_input('Time Precision')
year = st.number_input('Year')
month = st.number_input('Month')
day = st.number_input('Day')

# Create a dictionary from user inputs
input_data = {
    'ACTOR1': [actor1],
    'ACTOR2': [actor2],
    'ACTOR_DYAD_ID': [actor_dyad_id],
    'ADMIN1': [admin1],
    'ADMIN2': [admin2],
    'ADMIN3': [admin3],
    'COUNTRY': [country],
    'EVENT_TYPE': [event_type],
    'GEO_PRECISION': [geo_precision],
    'GWNO': [gwno],
    'INTER1': [inter1],
    'INTER2': [inter2],
    'INTERACTION': [interaction],
    'LATITUDE': [latitude],
    'LOCATION': [location],
    'LONGITUDE': [longitude],
    'TIME_PRECISION': [time_precision],
    'YEAR': [year],
    'month': [month],
    'day': [day]
}

# Create a DataFrame from the input data
input_df = pd.DataFrame(input_data)

# Preprocess the input data
input_df_transformed = label_encoder.transform(input_df.astype(str))
input_df_scaled = scaler.transform(input_df_transformed)

# Make predictions
prediction = model.predict(input_df_scaled)

# Map prediction to category
prediction_label = label_encoder.inverse_transform(prediction)[0]

# Display the prediction
st.subheader('Prediction:')

prediction_labels = ['Mild', 'Violent', 'Very Violent']
prediction_label = prediction_labels[prediction[0]]

st.write('Prediction:', prediction_label)