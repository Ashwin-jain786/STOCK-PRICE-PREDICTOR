import streamlit as st
import pandas as pd
from src.data_fetch import fetch_stock_data
from src.preprocessing import preprocess_data
from src.model_training import train_linear_regression, train_lstm_model, evaluate_model
from src.visualization import plot_stock_data, plot_predictions

# Streamlit app title
st.title("Stock Price Predictor")

def fetch_and_display_data(ticker, start_date, end_date):
    """Fetches stock data and displays it."""
    try:
        data = fetch_stock_data(ticker, start_date, end_date)

        if data.empty or data.isnull().values.any():
            st.error("No data found for the given ticker. Please check the ticker symbol and date range.")
            return None
        
        preprocessed_data = preprocess_data(data)        
        if preprocessed_data.empty or preprocessed_data.shape[0] < 200:
            st.error("Insufficient data for the selected date range. Please choose a different date range.")
            return None

        
        return preprocessed_data
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return None

def train_models(preprocessed_data):
    """Trains models and returns predictions."""
    X = preprocessed_data[['MA50', 'MA200', 'Normalized']]
    y = preprocessed_data['Close']
    
    # Validate that X and y are not empty before training
    if X.empty or y.empty:
        st.error("Input data is empty. Please check the data.")
        return None

    # Train Linear Regression model
    lr_model, lr_predictions = train_linear_regression(X, y)

    # Debugging: Check for non-numeric values in X
    st.write("X Data:", X)  # Debugging line
    st.write("y Data:", y)  # Debugging line

    # Train LSTM model (reshaping data for LSTM)
    X_lstm = X.values.reshape((X.shape[0], X.shape[1], 1))
    lstm_model = train_lstm_model(X_lstm, y)
    lstm_predictions = lstm_model.predict(X_lstm)

    # Debugging: Check predictions
    st.write("Linear Regression Predictions:", lr_predictions)  # Debugging line
    st.write("LSTM Predictions:", lstm_predictions)  # Debugging line

    return y, lr_predictions, lstm_predictions

# User input for stock ticker and date range
ticker = st.text_input("Enter Stock Ticker Symbol (e.g., AAPL):").strip().upper()

start_date = st.date_input("Start Date")
end_date = st.date_input("End Date")

if st.button("Fetch Data"):
    st.spinner("Fetching data...")
    preprocessed_data = fetch_and_display_data(ticker, start_date, end_date)

    try:
        y, lr_predictions, lstm_predictions = train_models(preprocessed_data)
    except Exception as e:
        st.error(f"An error occurred during model training: {e}")

    # Additional code for plotting and displaying metrics...
