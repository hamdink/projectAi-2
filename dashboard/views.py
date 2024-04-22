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

    # Your data visualization code here using Plotly or Matplotlib

    # For example:
    # fig = px.line(data, x='Date', y='AQI', title='Daily AQI Over Time')
    # graph = fig.to_html(full_html=False)

    context = {
        # Pass any data or variables you want to render in the template
        # For example, 'graph': graph
    }

    return render(request, 'index.html', context)
