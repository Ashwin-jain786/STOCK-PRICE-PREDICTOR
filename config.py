import os

# Load API keys from environment variables
ALPHA_VANTAGE_API_KEY = os.getenv('ALPHA_VANTAGE_API_KEY')

if not ALPHA_VANTAGE_API_KEY or ALPHA_VANTAGE_API_KEY == 'your_api_key_here':
    raise ValueError("API key is missing or invalid. Please set the ALPHA_VANTAGE_API_KEY environment variable.")
