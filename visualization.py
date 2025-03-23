import matplotlib.pyplot as plt
import os
import logging  # Importing logging module
import numpy as np  # Importing numpy module

def plot_stock_data(df, ticker):
    """
    Plots historical stock prices along with moving averages.
    
    Parameters:
    df (DataFrame): Stock data.
    ticker (str): Stock ticker symbol.
    """
    if df.empty or df.isnull().values.any():
        raise ValueError("DataFrame is empty. Cannot plot stock data.")
    
    plt.figure(figsize=(14, 7))
    plt.plot(df['Close'], label='Close Price', color='blue')
    plt.plot(df['MA50'], label='50-Day Moving Average', color='orange')
    plt.plot(df['MA200'], label='200-Day Moving Average', color='red')
    plt.plot(df['EMA50'], label='50-Day EMA', color='green')
    plt.plot(df['EMA200'], label='200-Day EMA', color='purple')
    plt.title(f'{ticker} Stock Price History')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.grid()
    
    # Save the plot as an image
    image_path = f"{ticker}_stock_price_history.png"
    plt.savefig(image_path)
    logging.info(f"Saved plot as {image_path}")
    
    plt.show()

def plot_predictions(y_true, y_pred):
    """
    Plots actual vs predicted stock prices.
    
    Parameters:
    y_true (array): True stock prices.
    y_pred (array): Predicted stock prices.
    """
    if len(y_true) == 0 or len(y_pred) == 0 or np.isnan(y_true).any() or np.isnan(y_pred).any():
        raise ValueError("True and predicted values cannot be empty.")
    
    plt.figure(figsize=(14, 7))
    plt.plot(y_true, label='Actual Prices', color='blue')
    plt.plot(y_pred, label='Predicted Prices', color='orange')
    plt.title('Actual vs Predicted Stock Prices')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.grid()
    
    # Save the plot as an image
    plt.savefig('actual_vs_predicted_stock_prices.png')
    logging.info("Saved plot as actual_vs_predicted_stock_prices.png")
    
    plt.show()
