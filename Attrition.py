import pandas as pd
import os
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import xgboost as xgb
import pickle

#Load Dataset
base_dir = os.path.dirname(__file__)
data_path = os.path.join(base_dir, "Data", "employee_attrition.csv")
df = pd.read_csv(data_path)

#Features and Target
features = [
    'MonthlyIncome', 'Age', 'JobSatisfaction', 'YearsAtCompany', 'OverTime',
    'JobRole', 'EnvironmentSatisfaction', 'JobLevel', 'TotalWorkingYears'
]
X = df[features].copy()

#Target Variable
y = df['Attrition'].map({'Yes': 1, 'No': 0})

#Preprocessing: Encode categorical columns
X['OverTime'] = X['OverTime'].map({'Yes': 1, 'No': 0})

categorical_features = ['JobRole']
numerical_features = [
    'MonthlyIncome', 'Age', 'JobSatisfaction', 'YearsAtCompany',
    'OverTime', 'EnvironmentSatisfaction', 'JobLevel', 'TotalWorkingYears'
]

#Preprocessing Pipeline
preprocessor = ColumnTransformer(transformers=[
    ('num', StandardScaler(), numerical_features),
    ('cat', OneHotEncoder(drop='first'), categorical_features)
])

#Class Imbalance Handling
yes_count = sum(y == 1)
no_count = sum(y == 0)
scale_pos_weight = no_count / yes_count
print(f"Attrition Counts: No={no_count}, Yes={yes_count}, scale_pos_weight={scale_pos_weight:.2f}")

# XGBoost Classifier
xgb_clf = xgb.XGBClassifier(
    random_state=42,
    use_label_encoder=False,
    eval_metric='logloss',
    scale_pos_weight=scale_pos_weight
)

# GridSearchCV Hyperparameters
param_grid = {
    'classifier__n_estimators': [100, 200],
    'classifier__max_depth': [3, 5],
    'classifier__learning_rate': [0.05, 0.1],
    'classifier__subsample': [0.8, 1.0],
    'classifier__colsample_bytree': [0.8, 1.0]
}

# Pipeline: Preprocessing + Classifier
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', xgb_clf)
])

#Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# GridSearchCV
grid_search = GridSearchCV(
    pipeline, param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=1
)
grid_search.fit(X_train, y_train)

# Best Model and Evaluation
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)

print("\n Best Parameters:", grid_search.best_params_)
print("\n Classification Report:\n", classification_report(y_test, y_pred))
print(" Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Save Final Model in 'Model' Folder
model_dir = os.path.join(base_dir, "Model")
os.makedirs(model_dir, exist_ok=True)
model_path = os.path.join(model_dir, "best_xgb_model.pkl")
with open(model_path, 'wb') as f:
    pickle.dump(best_model, f)