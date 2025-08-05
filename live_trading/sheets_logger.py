import gspread
from google.oauth2.service_account import Credentials
from datetime import datetime
import json
import os

class GoogleSheetsLogger:
    def __init__(self, sheet_url: str):
        # Setup Google Sheets connection
        scope = ['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive']
        
        # Use environment variable for credentials
        creds_json = os.getenv('GOOGLE_CREDENTIALS_JSON')
        if creds_json:
            creds = Credentials.from_service_account_info(json.loads(creds_json), scopes=scope)
        else:
            # Fallback to file (for local testing)
            creds = Credentials.from_service_account_file('credentials.json', scopes=scope)
            
        self.client = gspread.authorize(creds)
        
        # Open the sheet
        self.sheet = self.client.open_by_url(sheet_url).sheet1
        
        # Initialize headers if sheet is empty
        self.init_headers()
        
    def init_headers(self):
        """Initialize headers if sheet is empty"""
        if not self.sheet.get_all_values():
            headers = [
                'timestamp', 'instrument', 'action', 'units', 'price', 'order_id',
                'setup_time', 'entry_metrics', 'exit_reason', 'exit_price', 'exit_time',
                'duration_minutes', 'pnl_pips', 'pnl_usd', 'max_rr', 'win_loss',
                'stop_hit', 'tp_hit', 'tp_timing', 'max_drawdown_pips', 'max_drawdown_pct'
            ]
            self.sheet.append_row(headers)
        
    def log_trade_entry(self, trade_info: dict, analysis: dict):
        """Log a trade entry to Google Sheets"""
        row = [
            datetime.utcnow().isoformat(),
            trade_info['instrument'],
            'ENTRY',
            trade_info['units'],
            round(trade_info['price'], 5),  # Round to 5 decimal places
            trade_info['order_id'],
            analysis.get('setup_time', ''),
            json.dumps(analysis.get('metrics', {})),
            '',  # exit_reason
            '',  # exit_price
            '',  # exit_time
            '',  # duration_minutes
            '',  # pnl_pips
            '',  # pnl_usd
            '',  # max_rr
            '',  # win_loss
            '',  # stop_hit
            '',  # tp_hit
            '',  # tp_timing
            '',  # max_drawdown_pips
            '',  # max_drawdown_pct
            ''   # max_profit_before_stop_pips
        ]
        self.sheet.append_row(row)
        print(f"Trade entry logged to Google Sheets: {trade_info['instrument']}")
        
    def log_trade_exit(self, order_id: str, exit_data: dict):
        """Log a trade exit to Google Sheets"""
        # Find the row with this order_id and update it
        all_values = self.sheet.get_all_values()
        for i, row in enumerate(all_values[1:], start=2):  # Skip header row
            if row[5] == order_id:  # order_id is in column 6
                # Update exit data with 5 decimal place rounding
                update_range = f"J{i}:Z{i}"  # Columns J-Z for exit data
                exit_row = [
                    exit_data.get('exit_reason', ''),
                    round(exit_data.get('exit_price', 0), 5),  # Round to 5 decimal places
                    exit_data.get('exit_time', ''),
                    round(exit_data.get('duration_minutes', 0), 5),  # Round to 5 decimal places
                    round(exit_data.get('pnl_pips', 0), 5),  # Round to 5 decimal places
                    round(exit_data.get('pnl_usd', 0), 5),  # Round to 5 decimal places
                    round(exit_data.get('max_rr', 0), 5),  # Round to 5 decimal places
                    exit_data.get('win_loss', ''),
                    exit_data.get('stop_hit', ''),
                    exit_data.get('tp_hit', ''),
                    round(exit_data.get('tp_timing', 0), 5),  # Round to 5 decimal places
                    round(exit_data.get('max_drawdown_pips', 0), 5),  # Round to 5 decimal places
                    round(exit_data.get('max_drawdown_pct', 0), 5),  # Round to 5 decimal places
                    round(exit_data.get('max_profit_before_stop_pips', 0), 5)  # Round to 5 decimal places
                ]
                self.sheet.update(update_range, [exit_row])
                print(f"Trade exit logged to Google Sheets: {order_id}")
                break 
