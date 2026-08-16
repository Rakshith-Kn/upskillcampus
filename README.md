# Smart City Traffic Forecasting

## Overview

This project develops a machine learning based system for forecasting traffic patterns at different city junctions.

The system uses historical traffic data to identify traffic patterns and predict the expected number of vehicles.

## Problem Statement

Traffic congestion creates delays, increases fuel consumption and affects efficient city transportation.

The objective of this project is to use historical traffic data to forecast vehicle counts at different junctions and support better traffic planning.

## Objectives

- Analyze historical traffic patterns.
- Identify peak traffic periods.
- Study traffic variation across junctions.
- Develop machine learning models for traffic forecasting.
- Compare different regression models.
- Provide traffic predictions through an interactive application.

## Dataset

The dataset contains historical traffic observations including:

- Date and time
- Junction
- Vehicle count
- Record identifier

## Methodology

1. Data collection
2. Data cleaning
3. Feature engineering
4. Exploratory data analysis
5. Feature selection
6. Model training
7. Model evaluation
8. Prediction
9. Streamlit application

## Machine Learning Models

The project compares:

- Linear Regression
- Decision Tree Regressor
- Random Forest Regressor
- Gradient Boosting Regressor

## Evaluation Metrics

Models are evaluated using:

- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- R² Score

## Features

The model uses:

- Junction
- Year
- Month
- Day
- Hour
- Day of Week
- Weekend indicator
- Peak-hour indicator
- Previous traffic values
- Rolling traffic statistics

## Application

A Streamlit application provides an interface for entering traffic conditions and generating predicted vehicle counts.

## Technologies

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Joblib
- Streamlit

## Future Scope

Possible improvements include:

- Real-time traffic data integration
- Weather information
- Holiday information
- Traffic signal optimization
- Live city map integration
- More advanced time-series models