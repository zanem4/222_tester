import time
import json
import pandas as pd
import numpy as np
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
        try:
            # Get market data
            df = self.get_market_data(instrument)
            print(f"Got {len(df)} data points for {instrument}")
            
            # Check if we have enough data
            if len(df) < 3:  # Need at least 3 bars for 222 pattern
                print(f"Not enough data for {instrument} (need at least 3 bars)")
                return {
                    'setup_time': None,
                    'metrics': None,
                    'current_price': None,
                    'instrument': instrument
                }
            
            # Convert to numpy arrays for your existing strategy
            # Convert datetime to numeric timestamps
            time_array = df['time'].astype(np.int64) // 10**9  # Convert to Unix timestamp
            time_array = time_array.values  # Convert to numpy array
            high = df['high'].values
            low = df['low'].values
            close = df['close'].values
            open_prices = df['open'].values
            
            print(f"Data ranges - High: {high.min():.5f} to {high.max():.5f}")
            print(f"Time array length: {len(time_array)}")
            if len(time_array) > 0:
                print(f"Time range: {time_array[0]} to {time_array[-1]}")
            
            # Use your existing strategy modules
            # For live trading, use the current timestamp
            current_time = time_array[-1] if len(time_array) > 0 else 0
            print(f"Current time: {current_time}")
            
            print("Calculating metrics...")
            try:
                metrics = calculate_metrics(current_time, time_array, high, low, close, open_prices)
                print(f"Metrics calculated successfully: {len(metrics)} values")
            except Exception as e:
                print(f"Error in calculate_metrics: {e}")
                return {
                    'setup_time': None,
                    'metrics': None,
                    'current_price': close[-1],
                    'instrument': instrument
                }
            
            print("Normalizing metrics...")
            try:
                # Unpack the metrics array into individual parameters
                # Based on calculate_metrics return: [setup_idx, sma, price_range, atr, body1_size, body2_size, body3_size, upwick1, downwick1, upwick2, downwick2, upwick3, downwick3, norm_sma, norm_price_range, norm_body1, norm_body2, norm_body3, norm_upwick1, norm_downwick1, norm_upwick2, norm_downwick2, norm_upwick3, norm_downwick3]
                
                # Extract the raw metrics (first 13 values)
                setup_idx = metrics[0]
                sma = metrics[1]
                price_range = metrics[2]
                atr = metrics[3]
                body1_size = metrics[4]
                body2_size = metrics[5]
                body3_size = metrics[6]
                upwick1 = metrics[7]
                downwick1 = metrics[8]
                upwick2 = metrics[9]
                downwick2 = metrics[10]
                upwick3 = metrics[11]
                downwick3 = metrics[12]
                
                # For live trading, we don't need price_delta, so use 0
                price_delta = 0.0
                
                norm_metrics = normalize_metrics(price_delta, sma, price_range, body1_size, body2_size, body3_size,
                                               upwick1, downwick1, upwick2, downwick2, upwick3, downwick3, atr)
                print(f"Normalized metrics successfully")
            except Exception as e:
                print(f"Error in normalize_metrics: {e}")
                return {
                    'setup_time': None,
                    'metrics': None,
                    'current_price': close[-1],
                    'instrument': instrument
                }
            
            print("Detecting setups...")
            try:
                # detect_setups expects raw price arrays, not normalized metrics
                setups = detect_setups(time_array, high, low, close, open_prices)
                print(f"Setups detected: {len(setups) if setups is not None else 0}")
            except Exception as e:
                print(f"Error in detect_setups: {e}")
                return {
                    'setup_time': None,
                    'metrics': None,
                    'current_price': close[-1],
                    'instrument': instrument
                }
            
            print("Applying filter...")
            try:
                # For live trading, we need to handle multiple setups properly
                if len(setups) > 0:
                    # The issue is that detect_setups found multiple setups in historical data
                    # But we only have metrics for the current time
                    # For live trading, we only care about the most recent setup
                    
                    # Get the most recent setup (last in the array)
                    most_recent_setup = setups[-1] if len(setups) > 0 else None
                    
                    if most_recent_setup is not None:
                        # Create a single setup array with just the most recent setup
                        recent_setups = np.array([most_recent_setup])
                        
                        # Create a metrics array with the current metrics for this setup
                        current_metrics = np.array([metrics])  # Shape: (1, 25)
                        
                        # Apply the constraints from your optimized parameters
                        constraints = {k: v for k, v in self.parameters.items() if k.startswith('const_')}
                        
                        filtered_setups, filtered_metrics = apply_filter(recent_setups, current_metrics, constraints)
                        print(f"Most recent setup: {most_recent_setup}")
                        print(f"Filtered setups: {len(filtered_setups) if filtered_setups is not None else 0}")
                    else:
                        filtered_setups = np.array([])
                        print(f"Filtered setups: 0 (no recent setup)")
                else:
                    filtered_setups = np.array([])
                    print(f"Filtered setups: 0 (no setups to filter)")
            except Exception as e:
                print(f"Error in apply_filter: {e}")
                return {
                    'setup_time': None,
                    'metrics': None,
                    'current_price': close[-1],
                    'instrument': instrument
                }
            
            # Get the most recent setup
            setup_time = filtered_setups[-1] if len(filtered_setups) > 0 else None
            
            return {
                'setup_time': setup_time,
                'metrics': norm_metrics,
                'current_price': close[-1],
                'current_high': high[-1],  # Add this
                'current_low': low[-1],    # Add this
                'instrument': instrument
            }
            
        except Exception as e:
            print(f"Error in analyze_market for {instrument}: {e}")
            import traceback
            traceback.print_exc()
            return {
                'setup_time': None,
                'metrics': None,
                'current_price': None,
                'instrument': instrument
            }
    
    def execute_trade(self, analysis: Dict) -> Dict:
        """Execute a trade based on strategy analysis"""
        if not analysis['setup_time']:
            return {'status': 'no_setup'}
        
        # Get REAL-TIME account balance before each trade
        account_info = self.client.get_account_info()
        current_balance = float(account_info['account']['balance'])
        account_currency = account_info['account']['currency']
        
        print(f"=== TRADE EXECUTION ===")
        print(f"Account balance: {current_balance} {account_currency}")
        print(f"Account currency: {account_currency}")
        
        # Calculate 1% risk
        risk_per_trade = 0.01  # 1%
        risk_amount = current_balance * risk_per_trade
        
        print(f"Risk per trade: {risk_per_trade:.2%}")
        print(f"Risk amount: {risk_amount:.2f} {account_currency}")
        
        # Calculate proper position size based on currency pair
        instrument = analysis['instrument']
        base_currency = instrument[:3]  # First 3 chars (NZD in NZD_USD)
        quote_currency = instrument[4:]  # Last 3 chars (USD in NZD_USD)
        
        print(f"Currency pair: {base_currency}/{quote_currency}")
        
        # Calculate stop loss distance based on your 222 pattern strategy
        price_range = analysis['metrics'][2] if analysis['metrics'] is not None else 0.001
        stop_loss_distance = price_range  # This is the price delta from 222 pattern
        
        print(f"Price range (222 pattern): {price_range:.5f}")
        print(f"Stop loss distance: {stop_loss_distance:.5f}")
        
        # Calculate pip value in account currency
        if quote_currency == account_currency:
            # Direct conversion (e.g., NZD_USD with USD account)
            pip_value = 0.10  # $0.10 per 1,000 units per pip
        else:
            # Need conversion (e.g., EUR_JPY with USD account)
            exchange_rate = self.client.get_exchange_rate(quote_currency, account_currency)
            pip_value = 0.10 * exchange_rate
            print(f"Exchange rate {quote_currency}/{account_currency}: {exchange_rate:.5f}")
        
        print(f"Pip value: {pip_value:.5f} {account_currency} per 1,000 units")
        
        # Calculate units based on risk and stop loss distance
        # Convert stop loss distance to pips
        stop_loss_pips = int(stop_loss_distance * 10000)  # Convert to pips
        stop_loss_pips = max(1, stop_loss_pips)  # Ensure at least 1 pip
        
        # Calculate units: risk_amount / (stop_loss_pips * pip_value_per_unit)
        # pip_value is per 1000 units, so divide by 1000
        units = int(risk_amount / (stop_loss_pips * pip_value / 1000))
        
        print(f"Stop loss pips: {stop_loss_pips}")
        print(f"Calculated units: {units}")
        print(f"Expected risk: {abs(units) * stop_loss_pips * pip_value / 1000:.2f} {account_currency}")

        # Ensure reasonable position size for margin requirements
        max_units = int(current_balance * 0.1)  # Max 10% of balance as units
        if abs(units) > max_units:
            units = max_units if units > 0 else -max_units
            print(f"Position size capped at {max_units} units for margin safety")
        
        # Ensure minimum position size
        if abs(units) < 1000:
            units = 1000 if units > 0 else -1000
            print(f"Position size set to minimum {1000} units")
        
        print(f"Final units: {units}")
        
        # Determine trade direction from setup time sign
        # Positive setup time = long, negative = short
        is_long = analysis['setup_time'] > 0
        side = "buy" if is_long else "sell"
        
        print(f"Trade direction: {side}")
        print(f"Expected risk: {abs(units) * stop_loss_pips * pip_value / 1000:.2f} {account_currency}")
        
        # Calculate order price based on 222 pattern strategy
        # Get the setup bar data (most recent completed bar)
        setup_high = analysis.get('current_high', analysis['current_price'])
        setup_low = analysis.get('current_low', analysis['current_price'])
        
        # Determine trade direction from setup time sign
        # Positive setup time = long, negative = short
        is_long = analysis['setup_time'] > 0
        side = "buy" if is_long else "sell"
        
        if is_long:
            # Long trade: place limit order at high of setup bar
            order_price = setup_high
            print(f"Long setup: placing limit buy at {order_price:.5f} (high of setup bar)")
        else:
            # Short trade: place limit order at low of setup bar
            order_price = setup_low
            print(f"Short setup: placing limit sell at {order_price:.5f} (low of setup bar)")
        
        # Place the LIMIT order (not market FOK)
        try:
            order_result = self.client.place_limit_order(
                instrument=analysis['instrument'],
                units=units,
                price=order_price,
                side=side
            )
            
            print(f"Limit order placed: {order_result}")
            
            # Check if order was immediately filled
            if order_result.get('orderFillTransaction'):
                # Convert numpy arrays to lists for JSON serialization
                serializable_analysis = {
                    'setup_time': int(analysis['setup_time']) if analysis['setup_time'] is not None else None,
                    'current_price': float(analysis['current_price']) if analysis['current_price'] is not None else None,
                    'instrument': analysis['instrument'],
                    'metrics': [float(x) for x in analysis['metrics']] if analysis['metrics'] is not None else None
                }
                
                trade_info = {
                    'order_id': order_result['orderFillTransaction']['id'],
                    'instrument': analysis['instrument'],
                    'units': units,
                    'price': float(order_result['orderFillTransaction']['price']),
                    'time': datetime.utcnow(),
                    'analysis': serializable_analysis  # Use serializable version
                }
                
                print(f"Order immediately filled at {trade_info['price']:.5f}")
                
                # Log the trade entry
                self.logger.log_trade_entry(trade_info, serializable_analysis)
                
                # Log to Google Sheets if available
                if self.sheets_logger:
                    try:
                        self.sheets_logger.log_trade_entry(trade_info, serializable_analysis)
                    except Exception as e:
                        print(f"Warning: Could not log to Google Sheets: {e}")
                
                self.trade_history.append(trade_info)
                return {'status': 'success', 'trade': trade_info}
            else:
                # Order placed but not filled yet
                print(f"Limit order placed but not filled yet. Order ID: {order_result.get('orderCreateTransaction', {}).get('id')}")
                return {'status': 'pending', 'order': order_result}
                
        except Exception as e:
            return {'status': 'error', 'error': str(e)}
    
    def monitor_positions(self):
        """Monitor and manage open positions"""
        try:
            positions = self.client.get_positions()
            
            for position in positions.get('positions', []):
                instrument = position['instrument']
                long_units = int(position['long']['units']) if position['long']['units'] != '0' else 0
                short_units = int(position['short']['units']) if position['short']['units'] != '0' else 0
                
                if long_units == 0 and short_units == 0:
                    continue
                
                units = long_units if long_units > 0 else short_units
                is_long = long_units > 0
                
                # Get current price
                price_data = self.client.get_current_price(instrument)
                current_price = float(price_data['prices'][0]['bids'][0]['price'])
                
                # Get position details
                position_id = position['id']
                entry_price = float(position['long']['averagePrice'] if is_long else position['short']['averagePrice'])
                
                print(f"Monitoring position: {instrument} {units} @ {entry_price}, current: {current_price}")
                
                # Calculate stop loss level based on 222 pattern
                # For now, use a simple ATR-based stop (you'll need to implement the actual 222 pattern stop)
                stop_distance = 0.005  # 50 pips as placeholder
                if is_long:
                    stop_level = entry_price - stop_distance
                    if current_price <= stop_level:
                        print(f"Stop loss hit for {instrument} long position")
                        self.close_position(instrument, units)
                else:
                    stop_level = entry_price + stop_distance
                    if current_price >= stop_level:
                        print(f"Stop loss hit for {instrument} short position")
                        self.close_position(instrument, units)
                
                # Update trade status in logger
                self.logger.update_trade_status(position_id, current_price, datetime.utcnow())
                
        except Exception as e:
            print(f"Error monitoring positions: {e}")

    def close_position(self, instrument: str, units: int):
        """Close a position"""
        try:
            result = self.client.close_position(instrument, units)
            print(f"Position closed: {result}")
            
            # Log the exit
            if hasattr(self, 'logger'):
                # You'll need to implement exit logging
                pass
                
        except Exception as e:
            print(f"Error closing position: {e}")
    
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
