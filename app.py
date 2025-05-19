import pandas as pd
import numpy as np
import streamlit as st
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
import pickle

## Loading the trained model
model = tf.keras.models.load_model('model.h5')

## Load the encoders and decoders and scaled values
with open('labelencoder_gender.pkl', 'rb') as file:
    labelencoder_gender = pickle.load(file)

with open('onehot_encoder_geo.pkl', 'rb') as file:
    onehot_encoder_geo = pickle.load(file)

with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

## streamlit app
st.title("Customer Churn Prediction")

## User Input
credit_score = st.number_input('Credit Score')
geography = st.selectbox('Geography', onehot_encoder_geo.categories_[0])
gender = st.selectbox('Gender', labelencoder_gender.classes_)
age = st.slider('Age', 18, 92)
tenure = st.slider('Tenure', 0, 10)
balance = st.number_input('Balance')
num_of_products = st.slider('Number of Products', 1, 4)
has_cr_card = st.selectbox('Has Credit Card', [0,1])
is_active_member = st.selectbox('Is Active Member', [0,1])
estimated_salary = st.number_input('Estimated Salary')

## Preparing Input Data
input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender': [labelencoder_gender.transform([gender])[0]],
    'Age':[age],
    'Tenure':[tenure],
    'Balance': [balance],
    'NumOfProducts' : [num_of_products],
    'HasCrCard':[has_cr_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary]
})

## onehot encoder geography
geo_encoded = onehot_encoder_geo.transform([[geography]]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded, columns = onehot_encoder_geo.get_feature_names_out(['Geography']))

input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)

## Scaling input data
input_data_scaled = scaler.transform(input_data)

## Churn prediction
prediction = model.predict(input_data_scaled)
prediction_proba = prediction[0][0]

st.subheader("Prediction Result:")
if prediction_proba > 0.5:
    st.success("The customer is likely to churn.")
else:
    st.success("Customer will not churn.")

st.metric(label="Churn Probability", value=f"{prediction_proba * 100:.2f}%")