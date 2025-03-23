import pandas as pd

def preprocess_data(df):
    """
    Preprocesses the stock data by handling missing values and creating new features.
    
    Parameters:
    df (DataFrame): Raw stock data.
    
    Returns:
    DataFrame: Preprocessed stock data.
    """
    from sklearn.impute import SimpleImputer

    # Handle missing values using SimpleImputer with median strategy
    imputer = SimpleImputer(strategy='median')
    df[['Close']] = imputer.fit_transform(df[['Close']])

    
    # Create Moving Averages, ensuring sufficient data length
    if len(df) < 200:
        raise ValueError("Insufficient data for calculating 200-day moving average.")
    if len(df) < 50:
        raise ValueError("Insufficient data for calculating 50-day moving average.")

    df['MA50'] = df['Close'].rolling(window=50).mean()
    df['MA200'] = df['Close'].rolling(window=200).mean()
    
    # Create Exponential Moving Averages
    df['EMA50'] = df['Close'].ewm(span=50, adjust=False).mean()
    df['EMA200'] = df['Close'].ewm(span=200, adjust=False).mean()
    
    # Normalize the price data
    df['Normalized'] = (df['Close'] - df['Close'].min()) / (df['Close'].max() - df['Close'].min())
    
    # Ensure all relevant columns are numeric
    df['MA50'] = pd.to_numeric(df['MA50'], errors='coerce')
    df['MA200'] = pd.to_numeric(df['MA200'], errors='coerce')
    df['EMA50'] = pd.to_numeric(df['EMA50'], errors='coerce')
    df['EMA200'] = pd.to_numeric(df['EMA200'], errors='coerce')
    df['Normalized'] = pd.to_numeric(df['Normalized'], errors='coerce')

    # Check for NaN values after preprocessing and log the number of imputed values
    num_imputed = df[['Close']].isnull().sum().sum()
    if num_imputed > 0:
        logging.info(f"Imputed {num_imputed} missing values in 'Close' column.")

    if df.isnull().values.any():
        raise ValueError("Data contains NaN values after preprocessing.")
    
    return df
