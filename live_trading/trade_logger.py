import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import os
import csv

class LiveTradeLogger:
    def __init__(self, log_dir: str = "live_trading/logs"):
        """
        Initialize live trade logger
        
        Args:
            log_dir: Directory to store trade logs
        """
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        
        # Create session log file
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_log_path = os.path.join(log_dir, f"session_{self.session_id}.csv")
        
        # Initialize session log
        self.init_session_log()
        
        # Track active trades
        self.active_trades = {}
        self.completed_trades = []
        
    def init_session_log(self):
        """Initialize the session log CSV file"""
        headers = [
            'timestamp', 'instrument', 'action', 'units', 'price', 'order_id',
            'setup_time', 'entry_metrics', 'exit_reason', 'exit_price', 'exit_time',
            'duration_minutes', 'pnl_pips', 'pnl_usd', 'max_rr', 'win_loss',
            'stop_hit', 'tp1_hit', 'tp2_hit', 'tp3_hit', 'tp4_hit', 'tp5_hit',
            'tp1_timing', 'tp2_timing', 'tp3_timing', 'tp4_timing', 'tp5_timing',
            'max_drawdown_pips', 'max_drawdown_pct', 'max_profit_before_stop_pips'
        ]
        
        with open(self.session_log_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(headers)
    
    def log_trade_entry(self, trade_info: Dict, analysis: Dict):
        """Log a new trade entry"""
        timestamp = datetime.utcnow()
        
        # Create trade record
        trade_record = {
            'timestamp': timestamp,
            'instrument': trade_info['instrument'],
            'action': 'ENTRY',
            'units': trade_info['units'],
            'price': trade_info['price'],
            'order_id': trade_info['order_id'],
            'setup_time': analysis['setup_time'],
            'entry_metrics': json.dumps(analysis['metrics']),
            'exit_reason': None,
            'exit_price': None,
            'exit_time': None,
            'duration_minutes': None,
            'pnl_pips': None,
            'pnl_usd': None,
            'max_rr': None,
            'win_loss': None,
            'stop_hit': None,
            'tp1_hit': None,
            'tp2_hit': None,
            'tp3_hit': None,
            'tp4_hit': None,
            'tp5_hit': None,
            'tp1_timing': None,
            'tp2_timing': None,
            'tp3_timing': None,
            'tp4_timing': None,
            'tp5_timing': None,
            'max_drawdown_pips': None,
            'max_drawdown_pct': None,
            'max_profit_before_stop_pips': None
        }
        
        # Store active trade
        self.active_trades[trade_info['order_id']] = {
            'entry_record': trade_record,
            'entry_time': timestamp,
            'entry_price': trade_info['price'],
            'instrument': trade_info['instrument'],
            'units': trade_info['units'],
            'direction': 'long' if trade_info['units'] > 0 else 'short',
            'tp_levels': self.calculate_tp_levels(trade_info['price'], trade_info['units']),
            'stop_level': self.calculate_stop_level(trade_info['price'], trade_info['units']),
            'max_profit_pips': 0,
            'max_drawdown_pips': 0,
            'current_rr': 0
        }
        
        # Write to CSV
        self.write_trade_record(trade_record)
        
        print(f"Trade logged: {trade_info['instrument']} {trade_info['units']} @ {trade_info['price']}")
    
    def calculate_tp_levels(self, entry_price: float, units: int) -> List[float]:
        """Calculate take profit levels based on your strategy parameters"""
        # This should match your backtesting TP calculation
        # You'll need to implement this based on your strategy
        tp_multipliers = [1.0, 1.5, 2.0, 2.5, 3.0]  # From your parameters
        atr_value = 0.001  # You'll need to calculate this from market data
        
        if units > 0:  # Long trade
            return [entry_price + (multiplier * atr_value) for multiplier in tp_multipliers]
        else:  # Short trade
            return [entry_price - (multiplier * atr_value) for multiplier in tp_multipliers]
    
    def calculate_stop_level(self, entry_price: float, units: int) -> float:
        """Calculate stop loss level"""
        # This should match your backtesting stop calculation
        atr_value = 0.001  # You'll need to calculate this from market data
        
        if units > 0:  # Long trade
            return entry_price - atr_value
        else:  # Short trade
            return entry_price + atr_value
    
    def update_trade_status(self, order_id: str, current_price: float, current_time: datetime):
        """Update trade status with current price"""
        if order_id not in self.active_trades:
            return
        
        trade = self.active_trades[order_id]
        entry_price = trade['entry_price']
        direction = trade['direction']
        
        # Calculate current RR
        if direction == 'long':
            current_rr = (current_price - entry_price) / (entry_price - trade['stop_level'])
        else:
            current_rr = (entry_price - current_price) / (trade['stop_level'] - entry_price)
        
        trade['current_rr'] = current_rr
        
        # Update max profit and drawdown
        if direction == 'long':
            profit_pips = (current_price - entry_price) * 10000
        else:
            profit_pips = (entry_price - current_price) * 10000
        
        trade['max_profit_pips'] = max(trade['max_profit_pips'], profit_pips)
        trade['max_drawdown_pips'] = min(trade['max_drawdown_pips'], profit_pips)
    
    def log_trade_exit(self, order_id: str, exit_price: float, exit_reason: str):
        """Log a trade exit"""
        if order_id not in self.active_trades:
            return
        
        trade = self.active_trades[order_id]
        exit_time = datetime.utcnow()
        
        # Calculate trade metrics
        duration = (exit_time - trade['entry_time']).total_seconds() / 60  # minutes
        
        if trade['direction'] == 'long':
            pnl_pips = (exit_price - trade['entry_price']) * 10000
        else:
            pnl_pips = (trade['entry_price'] - exit_price) * 10000
        
        # Calculate win/loss and max RR
        win_loss = pnl_pips > 0
        max_rr = trade['current_rr']
        
        # Determine exit details
        stop_hit = exit_reason == 'stop_loss'
        tp_hits = self.calculate_tp_hits(trade, exit_price, exit_reason)
        tp_timings = self.calculate_tp_timings(trade, exit_time)
        
        # Create exit record
        exit_record = {
            'timestamp': exit_time,
            'instrument': trade['instrument'],
            'action': 'EXIT',
            'units': trade['units'],
            'price': exit_price,
            'order_id': order_id,
            'setup_time': trade['entry_record']['setup_time'],
            'entry_metrics': trade['entry_record']['entry_metrics'],
            'exit_reason': exit_reason,
            'exit_price': exit_price,
            'exit_time': exit_time,
            'duration_minutes': duration,
            'pnl_pips': pnl_pips,
            'pnl_usd': pnl_pips * 10,  # Approximate USD value (you'll need to calculate this properly)
            'max_rr': max_rr,
            'win_loss': win_loss,
            'stop_hit': stop_hit,
            'tp1_hit': tp_hits[0],
            'tp2_hit': tp_hits[1],
            'tp3_hit': tp_hits[2],
            'tp4_hit': tp_hits[3],
            'tp5_hit': tp_hits[4],
            'tp1_timing': tp_timings[0],
            'tp2_timing': tp_timings[1],
            'tp3_timing': tp_timings[2],
            'tp4_timing': tp_timings[3],
            'tp5_timing': tp_timings[4],
            'max_drawdown_pips': trade['max_drawdown_pips'],
            'max_drawdown_pct': (trade['max_drawdown_pips'] / trade['max_profit_pips'] * 100) if trade['max_profit_pips'] > 0 else 0,
            'max_profit_before_stop_pips': trade['max_profit_pips']
        }
        
        # Write to CSV
        self.write_trade_record(exit_record)
        
        # Move to completed trades
        self.completed_trades.append({
            'entry': trade['entry_record'],
            'exit': exit_record,
            'trade_data': trade
        })
        
        # Remove from active trades
        del self.active_trades[order_id]
        
        print(f"Trade exit logged: {trade['instrument']} @ {exit_price} - {exit_reason}")
    
    def calculate_tp_hits(self, trade: Dict, exit_price: float, exit_reason: str) -> List[bool]:
        """Calculate which take profit levels were hit"""
        tp_levels = trade['tp_levels']
        direction = trade['direction']
        
        hits = [False] * 5
        
        if exit_reason == 'take_profit':
            for i, tp_level in enumerate(tp_levels):
                if direction == 'long' and exit_price >= tp_level:
                    hits[i] = True
                elif direction == 'short' and exit_price <= tp_level:
                    hits[i] = True
        
        return hits
    
    def calculate_tp_timings(self, trade: Dict, exit_time: datetime) -> List[Optional[float]]:
        """Calculate timing for each take profit level"""
        # This is a simplified version - you'll need to track actual TP hit times
        # For now, return None for all (indicating not tracked)
        return [None] * 5
    
    def write_trade_record(self, record: Dict):
        """Write a trade record to the CSV file"""
        with open(self.session_log_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                record['timestamp'], record['instrument'], record['action'],
                record['units'], record['price'], record['order_id'],
                record['setup_time'], record['entry_metrics'], record['exit_reason'],
                record['exit_price'], record['exit_time'], record['duration_minutes'],
                record['pnl_pips'], record['pnl_usd'], record['max_rr'],
                record['win_loss'], record['stop_hit'], record['tp1_hit'],
                record['tp2_hit'], record['tp3_hit'], record['tp4_hit'],
                record['tp5_hit'], record['tp1_timing'], record['tp2_timing'],
                record['tp3_timing'], record['tp4_timing'], record['tp5_timing'],
                record['max_drawdown_pips'], record['max_drawdown_pct'],
                record['max_profit_before_stop_pips']
            ])
    
    def get_session_summary(self) -> Dict:
        """Get summary statistics for the current session"""
        if not self.completed_trades:
            return {'total_trades': 0}
        
        total_trades = len(self.completed_trades)
        winning_trades = sum(1 for trade in self.completed_trades if trade['exit']['win_loss'])
        win_rate = winning_trades / total_trades
        
        total_pnl_pips = sum(trade['exit']['pnl_pips'] for trade in self.completed_trades)
        total_pnl_usd = sum(trade['exit']['pnl_usd'] for trade in self.completed_trades)
        
        avg_rr = np.mean([trade['exit']['max_rr'] for trade in self.completed_trades])
        
        return {
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'win_rate': win_rate,
            'total_pnl_pips': total_pnl_pips,
            'total_pnl_usd': total_pnl_usd,
            'avg_rr': avg_rr,
            'session_id': self.session_id
        }
    
    def export_session_data(self, output_path: Optional[str] = None):
        """Export session data in the same format as backtesting"""
        if output_path is None:
            output_path = os.path.join(self.log_dir, f"session_{self.session_id}_export.csv")
        
        # Create export data in the same format as your backtesting results
        export_data = []
        
        for trade in self.completed_trades:
            entry = trade['entry']
            exit = trade['exit']
            
            export_row = {
                'symbol': entry['instrument'],
                'entry_time': entry['timestamp'],
                'exit_time': exit['exit_time'],
                'direction': 'long' if entry['units'] > 0 else 'short',
                'fill_price': entry['price'],
                'stop_level': trade['trade_data']['stop_level'],
                'trade_duration_bars': exit['duration_minutes'] / 5,  # Assuming 5-minute bars
                'exit_reason': exit['exit_reason'],
                'win_loss': exit['win_loss'],
                'max_rr': exit['max_rr'],
                'stop_hit': exit['stop_hit'],
                'tp1_hit': exit['tp1_hit'],
                'tp2_hit': exit['tp2_hit'],
                'tp3_hit': exit['tp3_hit'],
                'tp4_hit': exit['tp4_hit'],
                'tp5_hit': exit['tp5_hit'],
                'tp1_timing': exit['tp1_timing'],
                'tp2_timing': exit['tp2_timing'],
                'tp3_timing': exit['tp3_timing'],
                'tp4_timing': exit['tp4_timing'],
                'tp5_timing': exit['tp5_timing'],
                'max_drawdown_pips': exit['max_drawdown_pips'],
                'max_drawdown_pct': exit['max_drawdown_pct'],
                'max_profit_before_stop_pips': exit['max_profit_before_stop_pips']
            }
            
            export_data.append(export_row)
        
        # Save to CSV
        df = pd.DataFrame(export_data)
        df.to_csv(output_path, index=False)
        
        print(f"Session data exported to: {output_path}")
        return output_path 