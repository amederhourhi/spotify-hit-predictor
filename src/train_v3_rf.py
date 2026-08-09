# What: Import machine learning models (Linear Regression vs Random Forest) and split tools.
# Why: To test a non-linear tree-based ensemble against our linear baseline using a 3-way data split.
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

def train_v3_models(data_path: str = "../data/raw/synthetic_spotify_v2_messy.csv"):
    # 1. Load Data
    df = pd.read_csv(data_path)
    
    # Clean outlier durations (>0 and <1 hour)
    df = df[(df['duration_ms'] > 0) & (df['duration_ms'] < 3600000)]
    
    X = df.drop(columns=['popularity_score'])
    y = df['popularity_score']
    
    # 2. Train / Validation / Test Split (70% Train, 15% Val, 15% Test)
    # What: Two-step split to create three isolated subsets.
    # Why: Train fits model weights, Val evaluates hyperparameter choices, Test remains pristine until the end.
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=42
    )
    
    print(f"Data Splits -> Train: {len(X_train)} | Validation: {len(X_val)} | Test: {len(X_test)}\n")
    
    # 3. Build Preprocessing Blueprints
    numeric_features = ['tempo', 'danceability', 'energy', 'valence', 'duration_ms']
    categorical_features = ['genre']
    
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    preprocessor = ColumnTransformer(transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])
    
    # 4. Compare Models on Validation Set
    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest (100 Trees)": RandomForestRegressor(n_estimators=100, random_state=42)
    }
    
    print("=== Validation Set Comparison ===")
    pipelines = {}
    for name, model in models.items():
        pipe = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('regressor', model)
        ])
        
        # Fit on Training Data
        pipe.fit(X_train, y_train)
        
        # Evaluate on Validation Data
        val_preds = pipe.predict(X_val)
        val_mae = mean_absolute_error(y_val, val_preds)
        val_r2 = r2_score(y_val, val_preds)
        
        print(f"{name}:")
        print(f"  Val MAE: {val_mae:.2f}")
        print(f"  Val R²:  {val_r2:.4f}")
        
        pipelines[name] = pipe
    
    # 5. Final Evaluation on Vaulted Test Set
    # What: Run the best performing model on the unseen test set.
    best_model_name = "Random Forest (100 Trees)"
    best_pipe = pipelines[best_model_name]
    
    test_preds = best_pipe.predict(X_test)
    test_mae = mean_absolute_error(y_test, test_preds)
    test_r2 = r2_score(y_test, test_preds)
    
    print(f"\n=== Vaulted Test Set Evaluation ({best_model_name}) ===")
    print(f"Test MAE: {test_mae:.2f} points")
    print(f"Test R²:  {test_r2:.4f}")

if __name__ == "__main__":
    train_v3_models()