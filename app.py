import streamlit as st
import joblib
import numpy as np

# Load the trained model and preprocessing tools
model = joblib.load('model.pkl')
encoder = joblib.load('label_encoder.pkl')
scaler = joblib.load('scaler.pkl')

# Streamlit App UI
st.title("Customer Churn Prediction")
st.write("Enter customer details to predict churn.")

# Input fields
state = st.selectbox("State Code", ['KS', 'OH', 'NJ', 'OK', 'AL', 'MA', 'MO', 'LA', 'WV', 'IN', 'RI', 'IA', 'MT', 'NY',
 'ID', 'VT', 'VA', 'TX', 'FL', 'CO', 'AZ', 'SC', 'NE', 'WY', 'HI', 'IL', 'NH', 'GA',
 'AK', 'MD', 'AR', 'WI', 'OR', 'MI', 'DE', 'UT', 'CA', 'MN', 'SD', 'NC', 'WA', 'NM',
 'NV', 'DC', 'KY', 'ME', 'MS', 'TN', 'PA', 'CT', 'ND'])

area_code = st.selectbox("Area Code", ['area_code_415', 'area_code_408', 'area_code_510'])

# Transform categorical variables using the preloaded encoder
area_code = encoder.transform([area_code])[0]  # Convert single value into a list and get the first element

account_length = st.number_input("Account Length (normalized)", min_value=1, max_value=243, step=0.01)

voice_plan = st.selectbox("Voice Plan", ['no', 'yes'])
voice_plan = encoder.transform([voice_plan])[0]

voice_messages = st.number_input("Voice Messages (normalized)", min_value=0, max_value=52, step=1)

intl_plan = st.selectbox("International Plan", ['no', 'yes'])
intl_plan = encoder.transform([intl_plan])[0]

intl_mins = st.number_input("International Minutes (normalized)", min_value=0, max_value=20, step=1)

intl_calls = st.number_input("International Calls (normalized)", min_value=0, max_value=20, step=1)

day_mins = st.number_input("Day Minutes (normalized)", min_value=0, max_value=351.5, step=1)

day_calls = st.number_input("Day Calls (normalized)", min_value=0, max_value=165, step=1)

eve_mins = st.number_input("Evening Minutes (normalized)", min_value=0, max_value=363.7, step=1)

eve_calls = st.number_input("Evening Calls (normalized)", min_value=0, max_value=170, step=1)

night_mins = st.number_input("Night Minutes (normalized)", min_value=0, max_value=395, step=1)

night_calls = st.number_input("Night Calls (normalized)", min_value=0, max_value=175, step=1)

customer_calls = st.number_input("Customer Service Calls (normalized)", min_value=0, max_value=9, step=0.01)

# Convert inputs into a NumPy array
input_data = np.array([[state, area_code, account_length, voice_plan, voice_messages, intl_plan, intl_mins, intl_calls,
                        day_mins, day_calls, eve_mins, eve_calls, night_mins, night_calls, customer_calls]])

# Apply the preloaded scaler (use only transform, not fit_transform)
input_data = scaler.transform(input_data)

# Predict button
if st.button("Predict Churn"):
    prediction = model.predict(input_data)

    # Show result
    if prediction[0] == 1:
        st.error("This customer is likely to churn. ")
    else:
        st.success("This customer is not likely to churn.")
