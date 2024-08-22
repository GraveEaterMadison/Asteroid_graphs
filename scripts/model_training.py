from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import pandas as pd
import logging
import joblib
import data_analysis

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def train_model(df, target_column):
    """Train a linear regression model with proper scaling and error handling."""
    # Check if the target column exists
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' does not exist in the DataFrame")
    
    # Handle categorical columns by converting them to numeric if needed
    df = pd.get_dummies(df, drop_first=True)
    
    # Define features and target
    X = df.drop(columns=[target_column])
    y = df[target_column]

    # Handle case where target column might be empty or have insufficient data
    if y.empty or X.empty:
        raise ValueError("Feature or target data is empty")

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # Define model and hyperparameter grid
    model = LinearRegression()
    param_grid = {}  # Define hyperparameters if any

    # Perform Grid Search for hyperparameter tuning
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='r2')
    grid_search.fit(X_train, y_train)

    # Best model
    best_model = grid_search.best_estimator_

    # Predict and evaluate
    predictions = best_model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    logger.info(f"Model trained with MSE: {mse:.4f}, R^2: {r2:.4f}")

    # Save the trained model and scaler
    joblib.dump(best_model, 'best_model.pkl')
    joblib.dump(scaler, 'scaler.pkl')

    return best_model

if __name__ == "__main__":
    data_type = input("Enter the type of data to analyze (asteroids/exoplanets): ").strip().lower()
    df = data_analysis.load_data(data_type)
    
    target_column = input("Enter the target column for model training: ").strip()
    
    try:
        best_model = train_model(df, target_column)
        print(f"Model trained and saved successfully.")
    except Exception as e:
        logger.error(f"An error occurred during model training: {e}")
