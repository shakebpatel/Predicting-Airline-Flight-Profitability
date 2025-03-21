# Airline Profitability Prediction

## **Project Overview**
The **Airline Profitability Prediction** project aims to develop a machine learning model to forecast airline profitability based on various operational and financial factors. This model will help airline operators optimize decision-making by understanding key profitability drivers.

## **Dataset**
The dataset includes historical flight performance data with the following features:
- **Flight delays**
- **Aircraft utilization**
- **Turnaround time**
- **Load factor**
- **Fleet availability**
- **Maintenance downtime**
- **Fuel efficiency**
- **Revenue**
- **Operating costs**
- **Seasonal fluctuations** and more.

Dataset Link: [Click Here](https://docs.google.com/spreadsheets/d/1eALZhnY5bEJ4uCi9BCjN2fpx8jRIzwWo/edit?usp=sharing&ouid=109976760607215104976&rtpof=true&sd=true)

## **Objective**
- Build a high-performance ML model to predict airline profitability.
- Handle real-world uncertainties like seasonal fluctuations, inefficiencies, and cost variations.
- Ensure **interpretability** to provide actionable insights.

## **Installation**
### **Prerequisites**
Make sure you have Python installed. You can install the required libraries using:
```bash
pip install -r requirements.txt
```

### **Dependencies**
The project uses the following Python libraries:
```txt
pandas
numpy
matplotlib
seaborn
scikit-learn
xgboost
shap
```

## **Usage**
### **1. Load Data**
```python
import pandas as pd
# Load dataset
df = pd.read_csv("airline_profit_data.csv")
print(df.head())
```
### **2. Train Model**
```python
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

X = df.drop(columns=['Profit'])  # Features
y = df['Profit']  # Target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
```
### **3. Evaluate Model**
```python
from sklearn.metrics import mean_absolute_error, mean_squared_error

y_pred = model.predict(X_test)
print("MAE:", mean_absolute_error(y_test, y_pred))
print("MSE:", mean_squared_error(y_test, y_pred))
```

## **Model Explainability**
To understand the key factors influencing profitability, we use SHAP (SHapley Additive exPlanations):
```python
import shap
explainer = shap.Explainer(model)
shap_values = explainer(X_test)
shap.summary_plot(shap_values, X_test)
```

## **Contributing**
Feel free to fork this repository and contribute by submitting a pull request.

## **License**
This project is licensed under the MIT License.

---

## **How to Upload to GitHub**
### **Step 1: Initialize Git Repository**
```bash
git init
git add .
git commit -m "Initial commit"
```
### **Step 2: Link to GitHub Repository**
```bash
git remote add origin <https://github.com/aman7756068021/Predicting-Airline-Flight-Profitability/tree/aman7756068021/User-Authentication-System-with-Node.js-and-Express>
git branch -M main
git push -u origin main
```

