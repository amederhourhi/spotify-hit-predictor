# What: Import libraries for data manipulation and random selection.
# Why: We need pandas to load/save the CSV and numpy to randomly pick which rows to corrupt.
import pandas as pd
import numpy as np

def generate_messy_data(input_path: str = "../data/raw/synthetic_spotify_v1.csv", 
                        output_path: str = "../data/raw/synthetic_spotify_v2_messy.csv"):
    
    # What: Load the clean dataset from V1.
    # Why: We use our baseline as the foundation so we can see how messy data hurts the model.
    df = pd.read_csv(input_path)
    
    # What: Set random seed.
    # Why: Ensures we corrupt the exact same rows every time the script is run.
    np.random.seed(42)
    
    # What: Introduce Categorical Data (Genre).
    # Why: ML models only understand numbers. We will have to learn how to encode these later.
    # How: Randomly assign one of four genres to every track.
    genres = ['Pop', 'Rock', 'Hip-Hop', 'Classical']
    df['genre'] = np.random.choice(genres, size=len(df))
    
    # What: Inject Missing Values (NaNs) into 'danceability'.
    # Why: We will learn how to 'impute' (fill in) missing data without dropping entire rows.
    # How: Select 5% of the rows randomly and replace their score with np.nan.
    missing_indices = np.random.choice(df.index, size=int(len(df) * 0.05), replace=False)
    df.loc[missing_indices, 'danceability'] = np.nan
    
    # What: Inject Outliers / Invalid Data into 'duration_ms'.
    # Why: We will learn how to write preprocessing rules to cap or remove impossible values.
    # How: Make 10 random tracks have a negative duration, and 10 have massive 10-hour durations.
    outlier_neg_indices = np.random.choice(df.index, size=10, replace=False)
    df.loc[outlier_neg_indices, 'duration_ms'] = -9999.0
    
    outlier_pos_indices = np.random.choice(df.index, size=10, replace=False)
    df.loc[outlier_pos_indices, 'duration_ms'] = 36000000.0 # 10 hours
    
    # What: Save the corrupted dataset.
    # Why: This file will be the new starting point for our V2 preprocessing pipeline.
    df.to_csv(output_path, index=False)
    print(f"Success! Corrupted data saved to {output_path}")
    print(f"Number of missing 'danceability' values: {df['danceability'].isna().sum()}")

# What: Execution guard.
if __name__ == "__main__":
    generate_messy_data()