# src/train_lightgbm.py
import sqlite3
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from lightgbm import LGBMRegressor

def train_lgbm():
    db_path = "data/spotify_data.db"
    print(f"Connecting to database at {db_path}...")
    
    conn = sqlite3.connect(db_path)
    query = "SELECT danceability, energy, key, loudness, mode, speechiness, acousticness, instrumentalness, liveness, valence, tempo, is_explicit, genre, popularity_score FROM clean_tracks"
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    # 1. Feature Engineering & Encoding
    df = pd.get_dummies(df, columns=['genre'], drop_first=True)
    
    X = df.drop(columns=['popularity_score']).astype(float)
    y = df['popularity_score']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("\nTraining LightGBM Regressor model...")
    lgbm = LGBMRegressor(
        n_estimators=150,
        learning_rate=0.08,
        max_depth=6,
        random_state=42,
        n_jobs=-1
    )
    
    lgbm.fit(X_train, y_train)
    
    predictions = lgbm.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)
    
    print("\n=== LIGHTGBM PERFORMANCE EVALUATION ===")
    print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
    print(f"R-Squared ($R^2$ Score): {r2:.4f}")

if __name__ == "__main__":
    train_lgbm()