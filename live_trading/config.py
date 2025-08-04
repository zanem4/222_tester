import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# OANDA API Configuration
OANDA_CONFIG = {
    "api_token": os.getenv("OANDA_API_TOKEN"),
    "account_id": os.getenv("OANDA_ACCOUNT_ID"),
    "environment": "practice"  # Use "live" for real trading
}

# Trading Configuration
TRADING_CONFIG = {
    "instruments": [
        "EUR_USD", "GBP_USD", "USD_JPY", "AUD_USD", "USD_CHF", "USD_CAD", 
        "EUR_JPY", "EUR_GBP", "EUR_CHF", "GBP_JPY", "AUD_JPY", "AUD_CHF", 
        "AUD_CAD", "CAD_JPY", "CAD_CHF", "CHF_JPY", "NZD_USD", "NZD_JPY"
    ],
    "check_interval": 60,  # 1 minute
    "max_positions": 3,
    "risk_per_trade": 0.01,
    "max_daily_trades": 10,
    "session_hours": [2, 22],
} 

# Google Sheets Configuration
SHEETS_CONFIG = {
    "sheet_url": "https://docs.google.com/spreadsheets/d/1-ab7-Ne7ImcWX8uWN97ud5L0TngJZvE7yQRDZ6QhL6g/edit?pli=1&gid=0#gid=0"

} 
