import yfinance as yf
import pandas as pd
import time
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)

def fetch_stock_data(ticker, start_date, end_date, use_alpha_vantage=False):
    """
    Fetches historical stock data from Yahoo Finance or Alpha Vantage.
    
    Parameters:
    ticker (str): Stock ticker symbol.
    start_date (str): Start date for fetching data (YYYY-MM-DD).
    end_date (str): End date for fetching data (YYYY-MM-DD).
    use_alpha_vantage (bool): Flag to use Alpha Vantage instead of yfinance.
    
    Returns:
    DataFrame: Stock data as a Pandas DataFrame.
    """
    retries = 3
    for attempt in range(retries):
        try:
            if use_alpha_vantage:
                # Implement Alpha Vantage API call here
                import requests
                import os

                ALPHA_VANTAGE_API_KEY = os.getenv('ALPHA_VANTAGE_API_KEY')
                url = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={ticker}&apikey={ALPHA_VANTAGE_API_KEY}"
                response = requests.get(url)
                
                if response.status_code == 200:
                    data = response.json()
                    if "Error Message" in data:
                        raise ValueError("Invalid ticker symbol or API key.")
                    if "Time Series (Daily)" not in data:
                        raise ValueError("No data found for the given ticker.")
                else:
                    raise ValueError(f"Error fetching data from Alpha Vantage API. Status code: {response.status_code}. Please check your API key and usage limits.")

                stock_data = pd.DataFrame.from_dict(data["Time Series (Daily)"], orient='index')
                
                # Handle NaN values more gracefully
                if stock_data.isnull().values.any():
                    logging.warning("NaN values found in the fetched stock data. Attempting to fill NaNs with forward fill.")
                    stock_data.fillna(method='ffill', inplace=True)

                stock_data.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
                stock_data.index = pd.to_datetime(stock_data.index)
                stock_data = stock_data.astype(float)
                return stock_data

            else:
                stock_data = yf.download(ticker, start=start_date, end=end_date)
                
                # Handle NaN values more gracefully
                if stock_data.isnull().values.any():
                    logging.warning("NaN values found in the fetched stock data. Attempting to fill NaNs with forward fill.")
                    stock_data.fillna(method='ffill', inplace=True)

                return stock_data
        except Exception as e:
            logging.error(f"Error fetching data for {ticker}: {e}")
            time.sleep(2)  # Wait before retrying
    return pd.DataFrame()  # Return an empty DataFrame on error
