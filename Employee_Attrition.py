import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

#Load trained model
base_dir = os.path.dirname(__file__)
model_path = os.path.join(base_dir, "Model", "best_xgb_model.pkl")
model = pickle.load(open(model_path, 'rb'))

#Final feature columns (must match model training)
columns = ['MonthlyIncome', 'Age', 'JobSatisfaction', 'YearsAtCompany', 'OverTime',
           'JobRole', 'EnvironmentSatisfaction', 'JobLevel', 'TotalWorkingYears']

st.set_page_config(page_title="Employee Attrition Predictor", layout="centered")

st.title("Employee Attrition Predictor")
st.write("Predict whether an employee is likely to leave based on company data.")

#Input Fields
monthly_income = st.number_input("Monthly Income", min_value=1000, max_value=100000, value=3000, step=100)
age = st.number_input("Age", min_value=18, max_value=65, value=30, step=1)
job_satisfaction = st.slider("Job Satisfaction (1-4)", min_value=1, max_value=4, value=3)
years_at_company = st.slider("Years at Company", min_value=0, max_value=40, value=5)
overtime = st.radio("OverTime", options=['Yes', 'No'])

job_role = st.selectbox("Job Role", options=[
    'Sales Executive', 'Research Scientist', 'Laboratory Technician', 'Manufacturing Director',
    'Healthcare Representative', 'Manager', 'Sales Representative',
    'Research Director', 'Human Resources'
])

environment_satisfaction = st.slider("Environment Satisfaction (1-4)", min_value=1, max_value=4, value=3)
job_level = st.slider("Job Level (1-5)", min_value=1, max_value=5, value=2)
total_working_years = st.slider("Total Working Years", min_value=0, max_value=40, value=8)

#Prediction Logic
if st.button("Predict Attrition"):
    overtime_value = 1 if overtime == 'Yes' else 0

    input_data = pd.DataFrame([[monthly_income, age, job_satisfaction, years_at_company, overtime_value,
                                job_role, environment_satisfaction, job_level, total_working_years]],
                                columns=columns)

    prediction = model.predict(input_data)[0]

    if prediction == 1:
        st.error("Prediction: High Chance of Attrition (Likely to Leave).")
    else:
        st.success("Prediction: Low Chance of Attrition (Likely to Stay).")
