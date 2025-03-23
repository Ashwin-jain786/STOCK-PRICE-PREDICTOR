import numpy as np
import pandas as pd
import logging
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.impute import SimpleImputer

# Set up logging
logging.basicConfig(level=logging.INFO)

def train_linear_regression(X, y):
    """Train a linear regression model."""
    try:
        # Log shapes for debugging
        logging.info(f"Input X shape: {X.shape}, y shape: {y.shape}")

        # Data validation
        if X.empty or y.empty:
            raise ValueError("Input data is empty.")
        
        # Handle missing values using SimpleImputer with median strategy
        imputer = SimpleImputer(strategy='median')
        num_imputed = X.isnull().sum().sum()
        X = imputer.fit_transform(X)

        # Log the number of missing values imputed
        if num_imputed > 0:
            logging.info(f"Imputed {num_imputed} missing values in input features.")

        # Validate data types
        if not np.issubdtype(X.dtype, np.number):
            raise ValueError("X contains non-numeric values.")
        
        # Validate the shapes of input and target arrays
        if X.shape[0] != y.shape[0]:
            raise ValueError("The number of samples in X and y must be the same.")

        # Feature scaling
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

        # Model training
        model = LinearRegression()
        model.fit(X_train, y_train)

        # Predictions
        predictions = model.predict(X_test)

        # Log shapes for debugging
        logging.info(f"X shape: {X.shape}, y shape: {y.shape}, Predictions shape: {predictions.shape}")

        return model, predictions
    except ValueError as ve:
        logging.error(f"ValueError: {ve}")
    except Exception as e:
        logging.error(f"An error occurred: {e}")

def train_lstm_model(X, y):
    """Train an LSTM model."""
    try:
        # Log shapes and types of input data
        logging.info(f"Input X shape: {X.shape}, type: {type(X)}; Input y shape: {y.shape}, type: {type(y)}")

        
        # Ensure compatibility with both pandas and NumPy inputs
        X = np.asarray(X)
        y = np.asarray(y)

        # Handle missing values using SimpleImputer with median strategy
        imputer = SimpleImputer(strategy='median')
        X = imputer.fit_transform(X)

        # Reshape data for LSTM
        X_reshaped = X.reshape((X.shape[0], X.shape[1], 1))


        # Define LSTM model
        model = Sequential()
        model.add(LSTM(50, activation='relu', input_shape=(X_reshaped.shape[1], 1)))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mse')

        # Train the model
        model.fit(X_reshaped, y.values, epochs=200, verbose=0)

        return model
    except Exception as e:
        logging.error(f"An error occurred while training the LSTM model: {e}")

def evaluate_model(model, X_test, y_test):
    """Evaluate the model performance."""
    try:
        # Reshape X_test for LSTM
        X_test_reshaped = np.asarray(X_test).reshape((X_test.shape[0], X_test.shape[1], 1))
        predictions = model.predict(X_test_reshaped)

        mse = mean_squared_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)
        logging.info(f"Model Evaluation - MSE: {mse}, R^2: {r2}")
        return mse, r2
    except Exception as e:
        logging.error(f"An error occurred during model evaluation: {e}")

# Additional functions for other models can be added similarly
