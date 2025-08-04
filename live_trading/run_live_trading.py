#!/usr/bin/env python3
"""
Live Trading Script for Railway Deployment
"""

import os
import sys
import time
from datetime import datetime
from live_strategy_runner import LiveStrategyRunner
from config import OANDA_CONFIG, TRADING_CONFIG
from trade_logger import LiveTradeLogger

def main():
    """Main function to run live trading on Railway"""
    
    print(f"Starting live trading bot at {datetime.utcnow()}")
    print(f"Environment: {OANDA_CONFIG['environment']}")
    print(f"Instruments: {TRADING_CONFIG['instruments']}")
    
    # Validate configuration
    if not OANDA_CONFIG["api_token"]:
        print("ERROR: Please set OANDA_API_TOKEN environment variable")
        return
    
    if not OANDA_CONFIG["account_id"]:
        print("ERROR: Please set OANDA_ACCOUNT_ID environment variable")
        return
    
    # Initialize live strategy runner
    try:
        runner = LiveStrategyRunner(
            api_token=OANDA_CONFIG["api_token"],
            account_id=OANDA_CONFIG["account_id"],
            parameters_path="utils/parameters.json"
        )
    except Exception as e:
        print(f"Failed to initialize strategy runner: {e}")
        return
    
    # Test connection
    try:
        account_info = runner.client.get_account_info()
        
        # Handle different response formats
        if 'account' in account_info:
            account = account_info['account']
            account_name = account.get('name', 'Unknown Account')
            balance = account.get('balance', 'Unknown')
            currency = account.get('currency', 'Unknown')
        else:
            # Fallback if response structure is different
            account_name = account_info.get('name', 'Unknown Account')
            balance = account_info.get('balance', 'Unknown')
            currency = account_info.get('currency', 'Unknown')
            
        print(f"Connected to account: {account_name}")
        print(f"Balance: {balance}")
        print(f"Currency: {currency}")
    except Exception as e:
        print(f"Failed to connect to OANDA: {e}")
        print(f"Account info response: {account_info if 'account_info' in locals() else 'No response'}")
        return
    
    # Start live trading with error handling
    try:
        runner.run_live_trading(
            instruments=TRADING_CONFIG["instruments"],
            check_interval=TRADING_CONFIG["check_interval"]
        )
    except KeyboardInterrupt:
        print("\nStopping live trading...")
        # Export final session data
        runner.logger.export_session_data()
    except Exception as e:
        print(f"Error in live trading: {e}")
        # Export session data even on error
        runner.logger.export_session_data()

if __name__ == "__main__":
    main() 
