# views.py
from django.shortcuts import render
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import joblib

def your_view(request):
    # Load the data
    data = pd.read_csv("AQ_Data.csv")
    data.dropna(axis=0, inplace=True)
    data['Date'] = pd.to_datetime(data['Date'])
    data['Year'] = data['Date'].dt.year
    data['Month'] = data['Date'].dt.month
    data['Day'] = data['Date'].dt.day
    data['Weekday'] = data['Date'].dt.weekday
    data['WeekOfYear'] = data['Date'].dt.isocalendar().week

    # Select features and target
    feature_columns = ['PM2.5', 'PM10', 'NO', 'NO2', 'NOx', 'NH3', 'CO', 'SO2', 'O3', 'Benzene', 'Toluene', 'Xylene', 'Year', 'Month', 'Day', 'Weekday', 'WeekOfYear']
    X = data[feature_columns]
    y = data['AQI']

    # Standardizing the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Load the trained model
    filename = 'Random Forest_model.pkl'
    best_model = joblib.load(filename)

    # Predict AQI on the test set
    y_pred = best_model.predict(X_scaled)

    # Create a DataFrame of actual and predicted AQI
    actual_predicted_df = pd.DataFrame({'Actual': y, 'Predicted': y_pred})

    # Convert DataFrame to JSON for rendering in HTML
    actual_predicted_json = actual_predicted_df.to_json(orient='records')

    context = {
        'actual_predicted_json': actual_predicted_json
    }

    return render(request, 'index.html', context)

