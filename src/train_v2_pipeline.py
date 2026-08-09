# What: Import data manipulation and scikit-learn pipeline/preprocessing tools.
# Why: To professionally handle missing data, text encoding, and feature scaling in one smooth workflow.
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

def train_pipeline_model(data_path: str = "../data/raw/synthetic_spotify_v2_messy.csv"):
    # 1. Load Data
    df = pd.read_csv(data_path)
    
    # What: Manually remove impossible outliers.
    # Why: Scikit-learn pipelines are great at modifying columns, but bad at deleting rows.
    # How: Keep only rows where duration_ms is greater than 0 and less than 1 hour (3.6 million ms).
    initial_shape = df.shape[0]
    df = df[(df['duration_ms'] > 0) & (df['duration_ms'] < 3600000)]
    print(f"Removed {initial_shape - df.shape[0]} outlier rows.\n")
    
    # 2. Separate Features (X) and Target (y)
    X = df.drop(columns=['popularity_score'])
    y = df['popularity_score']
    
    # 3. Train/Test Split
    # What: Split data BEFORE preprocessing.
    # Why: CRITICAL to prevent data leakage. The model shouldn't know the median of the test set.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # What: Define feature groupings.
    # Why: We treat numbers and text differently.
    numeric_features = ['tempo', 'danceability', 'energy', 'valence', 'duration_ms']
    categorical_features = ['genre']
    
    # What: Build the blueprint for handling numerical data.
    # Why: Fills NaNs with the median value, then scales all numbers to have a mean of 0 and variance of 1.
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    # What: Build the blueprint for handling categorical (text) data.
    # Why: Fills any missing text with 'missing', then converts text to binary (1/0) columns.
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    # What: Combine the numerical and categorical blueprints into one master preprocessor.
    # Why: ColumnTransformer applies the right rules to the right columns automatically.
    preprocessor = ColumnTransformer(transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])
    
    # What: Create the final Machine Learning Pipeline.
    # Why: Chains the preprocessor directly into the Linear Regression model. 
    # How: When we call .fit(), data flows through imputation -> scaling/encoding -> model.
    model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', LinearRegression())
    ])
    
    # What: Train the entire pipeline.
    # Why: Learns the median/scaling values from X_train, transforms X_train, and fits the model.
    model_pipeline.fit(X_train, y_train)
    
    # What: Generate predictions on the test set.
    # Why: The pipeline automatically applies the exact same transformations to X_test before predicting.
    predictions = model_pipeline.predict(X_test)
    
    # Evaluate
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    
    print("=== V2 Pipeline Performance ===")
    print(f"Mean Absolute Error (MAE): {mae:.2f} points")
    print(f"R² Score:                  {r2:.4f}")

if __name__ == "__main__":
    train_pipeline_model()