import os
import kagglehub
import pandas as pd

def download_and_prepare_dataset():
    """Downloads the maharshipandya Spotify Tracks Dataset (114k tracks) via kagglehub."""
    print("Downloading dataset from Kaggle Hub...")
    path = kagglehub.dataset_download("maharshipandya/-spotify-tracks-dataset")
    print(f"Dataset downloaded to: {path}")

    csv_files = [f for f in os.listdir(path) if f.endswith(".csv")]
    if not csv_files:
        raise FileNotFoundError(f"No CSV found in {path}")
    csv_path = os.path.join(path, csv_files[0])
    print(f"Loading: {csv_path}")

    df = pd.read_csv(csv_path)
    print(f"Raw dataset shape: {df.shape}")

    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])

    before = len(df)
    df = df.drop_duplicates(subset="track_id", keep="first")
    print(f"Dropped {before - len(df)} duplicate track_id rows -> {len(df)} unique tracks")

    return df

if __name__ == "__main__":
    df = download_and_prepare_dataset()

    output_path = "data/raw/spotify_static_data.csv"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"\nSuccess! Saved {len(df)} unique tracks.")
    print(f"Data saved to {output_path}")
    print(f"\nColumns: {list(df.columns)}")
    print(f"\nPopularity stats:\n{df['popularity'].describe()}")