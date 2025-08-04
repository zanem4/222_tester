import time
import json
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import sys
import os

# Add the parent directory to the path to import your existing modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from oanda_client import OandaClient
from strategy.detect_setups import detect_setups
from strategy.calculate_metrics import calculate_metrics
from strategy.normalize_metrics import normalize_metrics
from strategy.apply_filter import apply_filter
from trade_logger import LiveTradeLogger
from sheets_logger import GoogleSheetsLogger
from config import SHEETS_CONFIG

class LiveStrategyRunner:
    def __init__(self, api_token: str, account_id: str, 
                 parameters_path: str = "utils/parameters.json"):
        """
        Initialize live strategy runner
        
        Args:
            api_token: OANDA API token
            account_id: OANDA account ID
            parameters_path: Path to your strategy parameters
        """
        self.client = OandaClient(api_token, account_id, "practice")
        self.parameters_path = parameters_path
        self.parameters = self.load_parameters()
        self.active_positions = {}
        self.trade_history = []
        
        # Initialize trade logger
        self.logger = LiveTradeLogger()
        
        # Initialize Google Sheets logger
        try:
            self.sheets_logger = GoogleSheetsLogger(SHEETS_CONFIG["sheet_url"])
            print("Google Sheets logger initialized successfully")
        except Exception as e:
            print(f"Warning: Could not initialize Google Sheets logger: {e}")
            self.sheets_logger = None
        
    def load_parameters(self) -> Dict:
        """Load strategy parameters"""
        with open(self.parameters_path, 'r') as f:
            return json.load(f)
    
    def get_market_data(self, instrument: str, lookback_hours: int = 24) -> pd.DataFrame:
        """Get recent market data for analysis"""
        # Calculate start time
        start_time = datetime.utcnow() - timedelta(hours=lookback_hours)
        
        # Get historical data
        df = self.client.get_historical_data(
            instrument=instrument,
            granularity="M5",  # 5-minute candles
            from_time=start_time.isoformat() + "Z"
        )
        
        return df
    
    def analyze_market(self, instrument: str) -> Dict:
        """Analyze current market conditions using your strategy"""
        # Get market data
        df = self.get_market_data(instrument)
        
        # Convert to numpy arrays for your existing strategy
        time_array = df['time'].values
        high = df['high'].values
        low = df['low'].values
        close = df['close'].values
        open_prices = df['open'].values  # Add this line
        
        # Use your existing strategy modules
        # You need to provide a setup_time for calculate_metrics
        # For live trading, you might want to use the current time or a recent timestamp
        current_time = time_array[-1] if len(time_array) > 0 else 0
        
        metrics = calculate_metrics(current_time, time_array, high, low, close, open_prices)
        norm_metrics = normalize_metrics(metrics, self.parameters)
        setups = detect_setups(norm_metrics, self.parameters)
        filtered_setups = apply_filter(setups, self.parameters)
        
        # Get the most recent setup
        setup_time = filtered_setups[-1] if len(filtered_setups) > 0 else None
        
        return {
            'setup_time': setup_time,
            'metrics': norm_metrics,
            'current_price': close[-1],
            'instrument': instrument
        }
    
    def execute_trade(self, analysis: Dict) -> Dict:
        """Execute a trade based on strategy analysis"""
        if not analysis['setup_time']:
            return {'status': 'no_setup'}
        
        # Calculate position size based on your risk management
        account_info = self.client.get_account_info()
        balance = float(account_info['account']['balance'])
        risk_per_trade = self.parameters.get('risk_per_trade', 0.01)
        risk_amount = balance * risk_per_trade
        
        # Calculate stop loss distance (you'll need to implement this based on your strategy)
        stop_distance_pips = 50  # Placeholder - implement based on your strategy
        stop_distance_price = stop_distance_pips / 10000  # Convert pips to price
        
        # Calculate units based on risk
        units = int(risk_amount / stop_distance_price)
        
        # Ensure minimum position size
        if abs(units) < 1000:
            units = 1000 if units > 0 else -1000
        
        # Determine trade direction from your strategy
        # This is a placeholder - implement based on your strategy logic
        side = "buy"  # or "sell" based on your analysis
        
        # Place the order
        try:
            order_result = self.client.place_market_order(
                instrument=analysis['instrument'],
                units=units,
                side=side
            )
            
            if order_result.get('orderFillTransaction'):
                trade_info = {
                    'order_id': order_result['orderFillTransaction']['id'],
                    'instrument': analysis['instrument'],
                    'units': units,
                    'price': float(order_result['orderFillTransaction']['price']),
                    'time': datetime.utcnow(),
                    'analysis': analysis
                }
                
                # Log the trade entry
                self.logger.log_trade_entry(trade_info, analysis)
                
                # Log to Google Sheets if available
                if self.sheets_logger:
                    try:
                        self.sheets_logger.log_trade_entry(trade_info, analysis)
                    except Exception as e:
                        print(f"Warning: Could not log to Google Sheets: {e}")
                
                self.trade_history.append(trade_info)
                return {'status': 'success', 'trade': trade_info}
            else:
                return {'status': 'failed', 'error': order_result}
                
        except Exception as e:
            return {'status': 'error', 'error': str(e)}
    
    def monitor_positions(self):
        """Monitor and manage open positions"""
        positions = self.client.get_positions()
        
        for position in positions.get('positions', []):
            instrument = position['instrument']
            units = int(position['long']['units']) if position['long']['units'] != '0' else int(position['short']['units'])
            
            # Get current price
            price_data = self.client.get_current_price(instrument)
            current_price = float(price_data['prices'][0]['bids'][0]['price'])
            
            # Update trade status in logger
            order_id = position.get('id')
            if order_id:
                self.logger.update_trade_status(order_id, current_price, datetime.utcnow())
            
            # Basic exit conditions (implement your actual exit logic)
            # Example: Close position if it's been open for more than 24 hours
            # You'll need to track entry time for each position
            
            # For now, just log the position
            print(f"Monitoring position: {instrument} {units} @ {current_price}")
    
    def run_live_trading(self, instruments: List[str], check_interval: int = 300):
        """
        Run live trading loop
        
        Args:
            instruments: List of instruments to trade
            check_interval: Time between market checks (seconds)
        """
        print(f"Starting live trading for instruments: {instruments}")
        print(f"Check interval: {check_interval} seconds")
        
        try:
            # Print session summary every hour
            last_summary_time = datetime.utcnow()
            
            while True:
                print(f"\n--- Market Check: {datetime.utcnow()} ---")
                
                # Monitor existing positions
                self.monitor_positions()
                
                # Check each instrument for new setups
                for instrument in instruments:
                    print(f"Analyzing {instrument}...")
                    
                    # Analyze market
                    analysis = self.analyze_market(instrument)
                    
                    # Execute trade if setup found
                    if analysis['setup_time']:
                        print(f"Setup found for {instrument}, executing trade...")
                        result = self.execute_trade(analysis)
                        print(f"Trade result: {result}")
                    else:
                        print(f"No setup for {instrument}")
                
                # Wait before next check
                print(f"Waiting {check_interval} seconds before next check...")
                time.sleep(check_interval)
                
                # Print session summary every hour
                if (datetime.utcnow() - last_summary_time).total_seconds() > 3600:
                    summary = self.logger.get_session_summary()
                    print(f"\n=== Session Summary ===")
                    print(f"Total trades: {summary['total_trades']}")
                    print(f"Win rate: {summary['win_rate']:.2%}")
                    print(f"Total PnL: {summary['total_pnl_usd']:.2f} USD")
                    print(f"Average RR: {summary['avg_rr']:.3f}")
                    last_summary_time = datetime.utcnow()
                
        except KeyboardInterrupt:
            print("\nStopping live trading...")
            
            # Export final session data
            self.logger.export_session_data()
            
            # Print final summary
            summary = self.logger.get_session_summary()
            print(f"\n=== Final Session Summary ===")
            print(f"Total trades: {summary['total_trades']}")
            print(f"Win rate: {summary['win_rate']:.2%}")
            print(f"Total PnL: {summary['total_pnl_usd']:.16f} USD")
            print(f"Average RR: {summary['avg_rr']:.16f}")
