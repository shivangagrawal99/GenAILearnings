# Gen AI Application Developer (Level-3 to level-5)-Evening Slot
# Shivang

# 3. Can you predict the employee attrition in an organization based on the following features.
# The features and the dataset are given below. Use a classification model with KNN algorithm
# Features:
    # Age: Age of the employee (numerical).
    # JobRole: The job role/position of the employee (categorical).
    # MonthlyIncome: Employee's monthly salary (numerical).
    # JobSatisfaction: A rating from 1 to 4 indicating the employee's satisfaction with the job (numerical).
    # YearsAtCompany: Number of years the employee has been at the company (numerical).
    # Attrition: Target label indicating whether the employee left the company (1 for attrition, 0 for no attrition)
# dataset: Datasets/EmployeeAttrition.csv
    # Age,JobRole,MonthlyIncome,JobSatisfaction,YearsAtCompany,Attrition
    # 29,Sales Executive,4800,3,4,1
    # ...
    # 26, Research Scientist,4500,3,2,1

# step 1: import the libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder

# step 2: reading the data from the employee_attrition.csv file
df = pd.read_csv('Datasets/EmployeeAttrition.csv')

# step 3: data preprocessing - handle categorical variable (JobRole)

# Convert JobRole to numerical values
le = LabelEncoder()
df['JobRole'] = le.fit_transform(df['JobRole'])

# step 4: separate features and target
X = df.drop('Attrition', axis=1).values
y = df['Attrition'].values

# step 5: split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# step 6: create and train KNN model
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

# step 7: evaluate the model
accuracy = knn.score(X_test, y_test)
print(f"Model Accuracy: {accuracy:.2f}")

# step 8: make predictions on new data
print("\nEnter employee details for attrition prediction:")
age = int(input("Age: "))
print("Available JobRoles:", le.classes_)
job_role = input("JobRole: ")
monthly_income = float(input("Monthly Income: "))
job_satisfaction = int(input("Job Satisfaction (1-4): "))
years_at_company = float(input("Years at Company: "))

# Convert job role to numerical
job_role_encoded = le.transform([job_role])[0]

# Make prediction
employee_data = [[age, job_role_encoded, monthly_income, job_satisfaction, years_at_company]]
prediction = knn.predict(employee_data)
probability = knn.predict_proba(employee_data)

print(f"\nPrediction: {'Attrition (1)' if prediction[0] == 1 else 'No Attrition (0)'}")
print(f"Probability - No Attrition: {probability[0][0]:.2f}, Attrition: {probability[0][1]:.2f}")
