# What: Import data manipulation and machine learning utilities.
# Why: Pandas loads the CSV; scikit-learn handles data splitting, modeling, and evaluation metrics.
# How: Import specific functions from sklearn to keep memory usage low and explicit.
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score

def train_baseline_model(data_path: str = "../data/raw/synthetic_spotify_v1.csv"):
    # What: Load raw synthetic dataset into a Pandas DataFrame.
    # Why: Brings raw numbers off disk into memory for preprocessing and modeling.
    # How: pd.read_csv parses the CSV structure into tabular format.
    df = pd.read_csv(data_path)
    print("--- Dataset Head ---")
    print(df.head(3))
    print("\n--- Summary Statistics ---")
    print(df.describe().T[['mean', 'std', 'min', 'max']])
    
    # What: Separate features (X) from target variable (y).
    # Why: ML algorithms require a clear distinction between inputs and what they are trying to predict.
    # How: Drop 'popularity_score' for X; select only 'popularity_score' for y.
    X = df.drop(columns=['popularity_score'])
    y = df['popularity_score']
    
    # What: Split data into training (80%) and testing (20%) sets.
    # Why: Prevents evaluation bias. We test model performance on unseen data to catch memorization.
    # How: train_test_split uses random_state=42 for exact reproducibility across runs.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"\nTraining set size: {X_train.shape[0]} samples")
    print(f"Testing set size:  {X_test.shape[0]} samples\n")
    
    # What: Initialize and train the Ordinary Least Squares Linear Regression model.
    # Why: Establishes our baseline model using a simple, highly interpretable algorithm.
    # How: fit() calculates optimal feature weights (coefficients) that minimize squared residual errors.
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # What: Generate predictions on the untouched test dataset.
    # Why: Evaluates how well the trained mathematical equation generalizes to unseen tracks.
    # How: predict() multiplies test feature values by learned weights and adds the intercept.
    predictions = model.predict(X_test)
    
    # What: Compute evaluation metrics comparing predictions against actual test labels.
    # Why: Quantifies model accuracy in absolute points (MAE/RMSE) and explained variance (R²).
    # How: Pass actual target values (y_test) and model outputs (predictions) into metric functions.
    mae = mean_absolute_error(y_test, predictions)
    rmse = root_mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    
    print("=== V1 Model Performance ===")
    print(f"Mean Absolute Error (MAE):     {mae:.2f} points")
    print(f"Root Mean Squared Error (RMSE): {rmse:.2f} points")
    print(f"R² Score:                       {r2:.4f}")
    
    # What: Inspect feature coefficients learned by the model.
    # Why: Shows which audio features had the strongest positive or negative influence on popularity.
    # How: Pair feature column names with model.coef_ array values.
    print("\n=== Learned Feature Coefficients ===")
    for feature, coef in zip(X.columns, model.coef_):
        print(f"{feature:15s}: {coef:+.4f}")

# What: Execution guard to ensure code only runs when directly executed.
# Why: Best practice for Python modules to prevent unintended execution during imports.
# How: Evaluates __name__ magic variable.
if __name__ == "__main__":
    train_baseline_model()