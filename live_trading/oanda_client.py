import requests
import json
import time
import pandas as pd
from datetime import datetime, timedelta
import os
from typing import Dict, List, Optional

class OandaClient:
    def __init__(self, api_token: str, account_id: str, environment: str = "practice"):
        """
        Initialize OANDA API client
        
        Args:
            api_token: Your OANDA API token
            account_id: Your OANDA account ID
            environment: "practice" for paper trading, "live" for live trading
        """
        self.api_token = api_token
        self.account_id = account_id
        self.environment = environment
        
        # Set base URL based on environment
        if environment == "practice":
            self.base_url = "https://api-fxpractice.oanda.com"
        else:
            self.base_url = "https://api-fxtrade.oanda.com"
        
        # Headers for API requests
        self.headers = {
            "Authorization": f"Bearer {api_token}",
            "Content-Type": "application/json"
        }
    
    def get_account_info(self) -> Dict:
        """Get account information"""
        url = f"{self.base_url}/v3/accounts/{self.account_id}"
        response = requests.get(url, headers=self.headers)
        return response.json()
    
    def get_instruments(self) -> List[str]:
        """Get list of tradeable instruments"""
        url = f"{self.base_url}/v3/accounts/{self.account_id}/instruments"
        response = requests.get(url, headers=self.headers)
        data = response.json()
        return [instrument['name'] for instrument in data['instruments']]
    
    def get_current_price(self, instrument: str) -> Dict:
        """Get current price for an instrument"""
        url = f"{self.base_url}/v3/accounts/{self.account_id}/pricing"
        params = {"instruments": instrument}
        response = requests.get(url, headers=self.headers, params=params)
        return response.json()
    
    def get_historical_data(self, instrument: str, granularity: str = "M5", 
                          count: int = 5000, from_time: Optional[str] = None) -> pd.DataFrame:
        """
        Get historical price data
        
        Args:
            instrument: Currency pair (e.g., "EUR_USD")
            granularity: Timeframe ("M1", "M5", "M15", "M30", "H1", "H4", "D")
            count: Number of candles to retrieve
            from_time: Start time (ISO format)
        """
        url = f"{self.base_url}/v3/instruments/{instrument}/candles"
        params = {
            "granularity": granularity,
            "count": count
        }
        
        if from_time:
            params["from"] = from_time
        
        response = requests.get(url, headers=self.headers, params=params)
        data = response.json()
        
        # Convert to DataFrame
        candles = []
        for candle in data['candles']:
            candles.append({
                'time': candle['time'],
                'open': float(candle['mid']['o']),
                'high': float(candle['mid']['h']),
                'low': float(candle['mid']['l']),
                'close': float(candle['mid']['c']),
                'volume': int(candle['volume'])
            })
        
        df = pd.DataFrame(candles)
        df['time'] = pd.to_datetime(df['time'])
        return df
    
    def place_market_order(self, instrument: str, units: int, side: str = "buy") -> Dict:
        """
        Place a market order
        
        Args:
            instrument: Currency pair
            units: Number of units (positive for buy, negative for sell)
            side: "buy" or "sell"
        """
        url = f"{self.base_url}/v3/accounts/{self.account_id}/orders"
        
        data = {
            "order": {
                "type": "MARKET",
                "instrument": instrument,
                "units": str(units),
                "timeInForce": "FOK",
                "positionFill": "DEFAULT"
            }
        }
        
        response = requests.post(url, headers=self.headers, json=data)
        return response.json()
    
    def place_limit_order(self, instrument: str, units: int, price: float, 
                         side: str = "buy") -> Dict:
        """Place a limit order"""
        url = f"{self.base_url}/v3/accounts/{self.account_id}/orders"
        
        data = {
            "order": {
                "type": "LIMIT",
                "instrument": instrument,
                "units": str(units),
                "price": str(price),
                "timeInForce": "GTC",
                "positionFill": "DEFAULT"
            }
        }
        
        response = requests.post(url, headers=self.headers, json=data)
        return response.json()
    
    def get_positions(self) -> Dict:
        """Get current positions"""
        url = f"{self.base_url}/v3/accounts/{self.account_id}/positions"
        response = requests.get(url, headers=self.headers)
        return response.json()
    
    def close_position(self, instrument: str, units: Optional[int] = None) -> Dict:
        """Close a position"""
        url = f"{self.base_url}/v3/accounts/{self.account_id}/positions/{instrument}/close"
        
        data = {}
        if units:
            data["units"] = str(units)
        
        response = requests.put(url, headers=self.headers, json=data)
        return response.json()
    
    def get_trades(self, instrument: Optional[str] = None, count: int = 50) -> Dict:
        """Get recent trades"""
        url = f"{self.base_url}/v3/accounts/{self.account_id}/trades"
        params = {"count": count}
        
        if instrument:
            params["instrument"] = instrument
        
        response = requests.get(url, headers=self.headers, params=params)
        return response.json() 