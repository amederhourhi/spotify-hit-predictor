# What: Import data manipulation libraries.
# Why: NumPy handles fast numerical generation; Pandas handles tabular data structures and CSV export.
# How: We import them using standard community aliases (np, pd).
import numpy as np
import pandas as pd
import os

def generate_clean_spotify_data(num_samples: int = 1000, output_path: str = "../data/raw/synthetic_spotify_v1.csv"):
    # What: Set a random seed.
    # Why: Ensures reproducibility. If reviewers run this script, they get the exact same numbers.
    # How: np.random.seed locks the random number generator to a specific starting point.
    np.random.seed(42)
    
    # What: Generate base audio features using uniform and normal distributions.
    # Why: Real Spotify features fall within specific ranges (e.g., valence is 0.0 to 1.0).
    # How: np.random.uniform creates random floats between a low and high bound.
    tempo = np.random.uniform(60.0, 200.0, num_samples)         # Beats per minute
    danceability = np.random.uniform(0.0, 1.0, num_samples)     # 0.0 to 1.0 scale
    energy = np.random.uniform(0.0, 1.0, num_samples)           # 0.0 to 1.0 scale
    valence = np.random.uniform(0.0, 1.0, num_samples)          # Musical positiveness
    duration_ms = np.random.uniform(120000, 300000, num_samples) # 2 to 5 minutes
    
    # What: Calculate the target variable (popularity_score).
    # Why: Our Linear Regression needs a mathematical pattern to learn.
    # How: We create a baseline score, add weighted features, and inject random noise to make it realistic.
    noise = np.random.normal(0, 5, num_samples) # Mean 0, standard deviation 5
    popularity_score = (
        10 +                             # Baseline popularity
        (danceability * 30) +            # High danceability heavily boosts popularity
        (energy * 25) +                  # High energy boosts popularity
        (valence * 15) -                 # Positiveness helps slightly
        (np.abs(tempo - 120) * 0.1) +    # Penalize tempos too far from standard pop (120 BPM)
        noise                            # Add random human unpredictability
    )
    
    # What: Clip the target variable to a realistic 0-100 scale.
    # Why: Popularity cannot logically exceed 100 or drop below 0.
    # How: np.clip forces any values outside the boundary back to the boundary limits.
    popularity_score = np.clip(popularity_score, 0, 100)
    
    # What: Structure the generated arrays into a DataFrame.
    # Why: DataFrames allow us to easily inspect the data and export it to a CSV file.
    # How: We pass a dictionary mapping column names to our NumPy arrays.
    df = pd.DataFrame({
        'tempo': tempo,
        'danceability': danceability,
        'energy': energy,
        'valence': valence,
        'duration_ms': duration_ms,
        'popularity_score': popularity_score
    })
    
    # What: Ensure the output directory exists and save the DataFrame to a CSV.
    # Why: Prevents crashing if the data/raw folder was accidentally deleted or misspelled.
    # How: os.makedirs creates the path if missing, and to_csv writes the file without row indices.
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Success! Generated {num_samples} records and saved to {output_path}")

# What: Standard Python execution guard.
# Why: Prevents the function from running automatically if this file is imported by another script later.
# How: Checks if the script is being run directly from the terminal.
if __name__ == "__main__":
    generate_clean_spotify_data()