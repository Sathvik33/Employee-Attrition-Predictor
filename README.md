**EMPLOYEE ATTRITION PREDICTOR**

**Project Overview**

This project is a machine learning application that predicts whether an employee is likely to leave the company based on multiple HR-related factors. The application uses XGBoost Classifier with GridSearchCV tuning, built into a Streamlit web interface for easy prediction.

**Objective**
Predict employee attrition (whether an employee will leave or stay).

Use key HR analytics features like income, satisfaction, tenure, overtime, job role to make predictions.

Provide an easy-to-use web interface for real-time predictions.

**Features Used**
Feature	Description
MonthlyIncome	Employee’s monthly income
Age	Age of the employee
JobSatisfaction	Job satisfaction rating (1-4)
YearsAtCompany	Number of years at current company
OverTime	Whether the employee does overtime (Yes/No)
JobRole	Role of the employee (e.g., Manager, Sales Executive)
EnvironmentSatisfaction	Workplace satisfaction rating (1-4)
JobLevel	Level/position in the company (1-5)
TotalWorkingYears	Total years of professional experience


**Tech Stack**

Python

XGBoost Classifier

GridSearchCV for hyperparameter tuning

Scikit-Learn for preprocessing

Streamlit for UI

Pandas & NumPy for data manipulation



**Results**
Achieved 80% accuracy using limited HR features.

Balanced the dataset using scale_pos_weight due to class imbalance.

Easy real-time prediction via Streamlit interface.


**Author**
Maru Sathvik Reddy
GitHub: Sathvik33
