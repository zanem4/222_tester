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
        self.pending_orders = {}  # Track pending orders for timeout checking
        
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
        """Get most recent market data for analysis"""
        # Get the most recent candles by omitting from_time (OANDA returns latest when only count is provided)
        df = self.client.get_historical_data(
            instrument=instrument,
            granularity="M1",  # 1-minute candles
            count=30  # last 30 completed bars
        )
        
        print(f"Retrieved {len(df)} bars for {instrument}")
        return df
    
    def analyze_market(self, instrument: str) -> Dict:
        """Analyze current market conditions using your strategy"""
        try:
            # Get market data - ONLY 30 BARS
            df = self.get_market_data(instrument)
            print(f"Got {len(df)} data points for {instrument}")
            
            # DEBUG: Check raw time data
            print(f"Raw time data sample: {df['time'].iloc[0]} (type: {type(df['time'].iloc[0])})")
            print(f"Raw time data sample: {df['time'].iloc[-1]} (type: {type(df['time'].iloc[-1])})")
            
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
            # FIX: Ensure proper time conversion
            if isinstance(df['time'].iloc[0], str):
                # If time is string, parse it first
                time_array = pd.to_datetime(df['time']).astype(np.int64) // 10**9
            else:
                # If already datetime, convert directly
                time_array = df['time'].astype(np.int64) // 10**9
            
            time_array = time_array.values  # Convert to numpy array
            high = df['high'].values
            low = df['low'].values
            close = df['close'].values
            open_prices = df['open'].values
            
            # DEBUG: Add current time comparison
            current_unix_time = int(datetime.utcnow().timestamp())
            print(f"Current UTC time: {current_unix_time}")
            print(f"Data ranges - High: {high.min():.5f} to {high.max():.5f}")
            print(f"Time array length: {len(time_array)}")
            if len(time_array) > 0:
                print(f"Time range: {time_array[0]} to {time_array[-1]}")
                print(f"Most recent bar time: {time_array[-1]}")
                print(f"Time difference from now: {time_array[-1] - current_unix_time} seconds")
                
                # Check if timestamps are in the future
                if time_array[-1] > current_unix_time:
                    print(f"WARNING: Most recent bar time is in the future!")
                    print(f"Expected: <= {current_unix_time}, Got: {time_array[-1]}")
            
            # CHECK FOR SETUP ON LAST 3 BARS ONLY
            # Use your detect_setups function on the entire dataset
            print("Detecting setups...")
            try:
                setups = detect_setups(time_array, high, low, close, open_prices)
                print(f"Setups detected: {len(setups) if setups is not None else 0}")
                
                # Debug: Print all setup times
                if len(setups) > 0:
                    print(f"All setup times: {setups}")
                    print(f"Last 3 bar times: {time_array[-3:]}")
            except Exception as e:
                print(f"Error in detect_setups: {e}")
                return {
                    'setup_time': None,
                    'metrics': None,
                    'current_price': None,
                    'instrument': instrument
                }
            
            # Check if there's a setup in the last 3 bars
            if len(setups) == 0:
                print(f"No setup found for {instrument}")
                return {
                    'setup_time': None,
                    'metrics': None,
                    'current_price': None,
                    'instrument': instrument
                }
            
            # Get the most recent setup
            most_recent_setup = setups[-1]
            setup_time = abs(most_recent_setup)  # Remove sign for now
            
            print(f"Most recent setup time: {setup_time}")
            print(f"Last 3 bar times: {time_array[-3:]}")
            
            # Check if this setup is in the last 3 bars
            setup_idx = np.where(time_array == setup_time)[0]
            if len(setup_idx) == 0:
                print(f"Setup not found in time array for {instrument}")
                return {
                    'setup_time': None,
                    'metrics': None,
                    'current_price': None,
                    'instrument': instrument
                }
            
            setup_idx = setup_idx[0]
            last_3_bars_start = len(time_array) - 3
            
            print(f"Setup index: {setup_idx}")
            print(f"Last 3 bars start index: {last_3_bars_start}")
            print(f"Total bars: {len(time_array)}")
            
            if setup_idx < last_3_bars_start:
                print(f"Setup found but not in last 3 bars for {instrument}")
                print(f"Setup is at index {setup_idx}, but we need index >= {last_3_bars_start}")
                return {
                    'setup_time': None,
                    'metrics': None,
                    'current_price': None,
                    'instrument': instrument
                }
            
            print(f"Setup found in last 3 bars for {instrument} at index {setup_idx}")
            
            # NOW calculate metrics using the setup time
            print("Calculating metrics...")
            try:
                metrics = calculate_metrics(most_recent_setup, time_array, high, low, close, open_prices)
                print(f"Metrics calculated successfully: {len(metrics)} values")
            except Exception as e:
                print(f"Error in calculate_metrics: {e}")
                return {
                    'setup_time': None,
                    'metrics': None,
                    'current_price': None,
                    'instrument': instrument
                }
            
            print("Normalizing metrics...")
            try:
                # Unpack the metrics array into individual parameters
                setup_idx_metric = metrics[0]
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
                    'current_price': None,
                    'instrument': instrument
                }
            
            print("Applying filter...")
            try:
                # Apply the constraints from your optimized parameters
                constraints = {k: v for k, v in self.parameters.items() if k.startswith('const_')}
                
                # FIX: Manual filtering for the single constraint we care about
                # We only have norm_price_range constraints: lower=0.0, upper=1.25
                
                # Get the norm_price_range value (index 2 in normalized metrics)
                norm_price_range_value = norm_metrics[2]  # norm_price_range is at index 2
                
                print(f"Checking norm_price_range constraint: value = {norm_price_range_value:.6f}")
                print(f"Constraints: lower = {constraints.get('const_norm_price_range_lower', 'N/A')}, upper = {constraints.get('const_norm_price_range_upper', 'N/A')}")
                
                # Check lower bound
                if norm_price_range_value < constraints.get('const_norm_price_range_lower', 0.0):
                    print(f"Setup filtered out: norm_price_range = {norm_price_range_value:.6f} < {constraints.get('const_norm_price_range_lower', 0.0):.6f}")
                    return {
                        'setup_time': None,
                        'metrics': None,
                        'current_price': None,
                        'instrument': instrument
                    }
                
                # Check upper bound
                if norm_price_range_value > constraints.get('const_norm_price_range_upper', 1.25):
                    print(f"Setup filtered out: norm_price_range = {norm_price_range_value:.6f} > {constraints.get('const_norm_price_range_upper', 1.25):.6f}")
                    return {
                        'setup_time': None,
                        'metrics': None,
                        'current_price': None,
                        'instrument': instrument
                    }
                
                print(f"Setup passed norm_price_range filter: {norm_price_range_value:.6f}")
                
            except Exception as e:
                print(f"Error in apply_filter: {e}")
                return {
                    'setup_time': None,
                    'metrics': None,
                    'current_price': None,
                    'instrument': instrument
                }
            # Setup passed all checks!
            print(f"Setup confirmed for {instrument}")
            
            return {
                'setup_time': most_recent_setup,  # Keep the sign for direction
                'metrics': norm_metrics,  # normalized metrics
                'price_range_raw': float(price_range),  # use raw price range for sizing
                'current_price': close[-1],
                'current_high': high[-1],
                'current_low': low[-1],
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
        print(f"Account balance: {current_balance:.5f} {account_currency}")
        print(f"Account currency: {account_currency}")
        
        # Calculate 1% risk
        risk_per_trade = 0.01  # 1%
        risk_amount = current_balance * risk_per_trade
        
        print(f"Risk per trade: {risk_per_trade:.2%}")
        print(f"Risk amount: {risk_amount:.5f} {account_currency}")
        
        # Calculate proper position size based on currency pair
        instrument = analysis['instrument']
        base_currency = instrument[:3]  # First 3 chars (NZD in NZD_USD)
        quote_currency = instrument[4:]  # Last 3 chars (USD in NZD_USD)
        
        print(f"Currency pair: {base_currency}/{quote_currency}")
        
        # Calculate stop loss distance based on your 222 pattern strategy
        # Use raw price range from analysis (absolute price units), fall back if missing
        price_range = analysis.get('price_range_raw') if analysis.get('price_range_raw') is not None else (
            float(analysis['metrics'][2]) if analysis.get('metrics') is not None else 0.001
        )
        stop_distance = price_range  # Use actual 222 pattern range
        tp_distance = price_range * 2.0  # 2.0RR target
        
        print(f"Price range (222 pattern): {price_range:.5f}")
        print(f"Stop loss distance: {stop_distance:.5f}")
        
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
        stop_loss_pips = int(stop_distance * 10000)  # Convert to pips
        stop_loss_pips = max(1, stop_loss_pips)  # Ensure at least 1 pip
        
        # Calculate units: risk_amount / (stop_loss_pips * pip_value_per_unit)
        # pip_value is per 1000 units, so divide by 1000
        units = int(risk_amount / (stop_loss_pips * pip_value / 1000))
        
        print(f"Stop loss pips: {stop_loss_pips}")
        print(f"Calculated units: {units}")
        print(f"Expected risk: {abs(units) * stop_loss_pips * pip_value / 1000:.5f} {account_currency}")

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
                    'current_price': round(float(analysis['current_price']), 5) if analysis['current_price'] is not None else None,  # Round to 5 decimal places
                    'instrument': analysis['instrument'],
                    'metrics': [round(float(x), 5) for x in analysis['metrics']] if analysis['metrics'] is not None else None  # Round to 5 decimal places
                }
                
                trade_info = {
                    'order_id': order_result['orderFillTransaction']['id'],
                    'instrument': analysis['instrument'],
                    'units': units,
                    'price': round(float(order_result['orderFillTransaction']['price']), 5),  # Round to 5 decimal places
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
                order_id = order_result.get('orderCreateTransaction', {}).get('id')
                print(f"Limit order placed but not filled yet. Order ID: {order_id}")
                
                # FIX: Track pending order with timestamp for timeout checking
                self.pending_orders[order_id] = {
                    'order_time': datetime.utcnow(),
                    'instrument': analysis['instrument'],
                    'units': units,
                    'price': order_price,
                    'side': side,
                    'analysis': analysis
                }
                
                return {'status': 'pending', 'order': order_result}
                
        except Exception as e:
            return {'status': 'error', 'error': str(e)}
    
    def check_pending_orders(self):
        """Check pending orders for timeout (10-bar limit)"""
        current_time = datetime.utcnow()
        orders_to_cancel = []
        
        for order_id, order_info in self.pending_orders.items():
            # Calculate time elapsed since order placement
            time_elapsed = current_time - order_info['order_time']
            elapsed_minutes = time_elapsed.total_seconds() / 60
            
            # 10 bars = 10 minutes (assuming 1-minute bars)
            max_wait_minutes = self.parameters.get('max_order_wait_bars', 10)
            
            if elapsed_minutes >= max_wait_minutes:
                print(f"Cancelling order {order_id} after {elapsed_minutes:.1f} minutes (timeout)")
                try:
                    # Cancel the order
                    self.client.cancel_order(order_id)
                    orders_to_cancel.append(order_id)
                except Exception as e:
                    print(f"Error cancelling order {order_id}: {e}")
        
        # Remove cancelled orders from tracking
        for order_id in orders_to_cancel:
            del self.pending_orders[order_id]

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
                
                # FIX: Check trade lifetime (50-bar limit)
                # Get position creation time from trade history
                position_age_minutes = 0
                for trade in self.trade_history:
                    if trade.get('order_id') == position_id:
                        position_age = datetime.utcnow() - trade['time']
                        position_age_minutes = position_age.total_seconds() / 60
                        break
                
                max_lifespan_minutes = self.parameters.get('max_trade_lifespan_bars', 50)
                
                if position_age_minutes >= max_lifespan_minutes:
                    print(f"Closing position {position_id} due to max lifespan ({position_age_minutes:.1f} minutes)")
                    self.close_position(instrument, units, "max_lifespan")
                    continue
                
                # Calculate stop loss and take profit levels
                # For now, use simple ATR-based levels
                stop_distance = 0.005  # 50 pips as placeholder
                tp_distance = 0.010    # 100 pips for 2.0RR
                
                if is_long:
                    stop_level = entry_price - stop_distance
                    tp_level = entry_price + tp_distance
                    
                    if current_price <= stop_level:
                        print(f"Stop loss hit for {instrument} long position")
                        self.close_position(instrument, units, "stop_loss")
                    elif current_price >= tp_level:
                        print(f"Take profit hit for {instrument} long position (2.0RR)")
                        self.close_position(instrument, units, "take_profit")
                else:
                    stop_level = entry_price + stop_distance
                    tp_level = entry_price - tp_distance
                    
                    if current_price >= stop_level:
                        print(f"Stop loss hit for {instrument} short position")
                        self.close_position(instrument, units, "stop_loss")
                    elif current_price <= tp_level:
                        print(f"Take profit hit for {instrument} short position (2.0RR)")
                        self.close_position(instrument, units, "take_profit")
                
                # Update trade status in logger
                self.logger.update_trade_status(position_id, current_price, datetime.utcnow())
                
        except Exception as e:
            print(f"Error monitoring positions: {e}")
    
    def close_position(self, instrument: str, units: int, exit_reason: str):
        """Close a position"""
        try:
            result = self.client.close_position(instrument, units)
            print(f"Position closed ({exit_reason}): {result}")
            
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
                
                # FIX: Check pending orders for timeout
                self.check_pending_orders()
                
                # Monitor existing positions
                self.monitor_positions()
                
                # Check each instrument for new setups
                for instrument in instruments:
                    print(f"Analyzing {instrument}...")
                    
                    # Analyze market
                    analysis = self.analyze_market(instrument)
                    
                    # ONLY execute trade if setup_time is not None
                    if analysis['setup_time'] is not None:
                        print(f"Setup found for {instrument}, executing trade...")
                        trade_result = self.execute_trade(analysis)
                        print(f"Trade result: {trade_result}")
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
                    if 'win_rate' in summary:  # FIX: Check if win_rate exists
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
            if 'win_rate' in summary:  # FIX: Check if win_rate exists
                print(f"Win rate: {summary['win_rate']:.2%}")
                print(f"Total PnL: {summary['total_pnl_usd']:.16f} USD")
                print(f"Average RR: {summary['avg_rr']:.16f}")
