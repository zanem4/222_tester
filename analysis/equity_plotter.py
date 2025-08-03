import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import sys
import os
from datetime import datetime, timedelta
import random

def calculate_equity_curve(csv_path: str, starting_balance: float = 100000, risk_per_trade: float = 0.01, commission_per_lot: float = 7.0):
    """
    Calculate equity curve from trade results with position tracking
    
    Args:
        csv_path: Path to the CSV file with trade results
        starting_balance: Starting account balance (default 100,000)
        risk_per_trade: Risk per trade as fraction of balance (default 1%)
        commission_per_lot: Commission per standard lot (100,000 units) in USD (default $7)
    
    Returns:
        DataFrame with equity curve data and position tracking
    """
    print(f"Loading CSV: {csv_path}")
    df = pd.read_csv(csv_path)
    
    if 'max_rr' not in df.columns:
        raise ValueError("CSV must contain 'max_rr' column")
    
    if 'price_range' not in df.columns:
        raise ValueError("CSV must contain 'price_range' column")
    
    # Check if we have timestamp data
    has_timestamps = 'entry_time' in df.columns and 'exit_time' in df.columns
    
    if has_timestamps:
        print("Timestamp data found - tracking simultaneous positions")
        # Parse timestamps as datetime (not Unix seconds)
        df['entry_time'] = pd.to_datetime(df['entry_time'])
        df['exit_time'] = pd.to_datetime(df['exit_time'])
    else:
        print("No timestamp data - using trade sequence for position tracking")
    
    print(f"Total trades: {len(df)}")
    print(f"Starting balance: ${starting_balance:,.2f}")
    print(f"Risk per trade: {risk_per_trade*100:.2f}% of current balance")
    print(f"Commission per lot: ${commission_per_lot:.2f}")
    
    # Initialize equity tracking
    current_balance = starting_balance
    equity_curve = []
    open_positions = []  # Track open positions
    max_simultaneous_positions = 0
    daily_exposure = {}  # Track daily risk exposure
    
    # Process each trade
    for idx, trade in enumerate(df.itertuples(index=False), 1):
        max_rr = getattr(trade, 'max_rr', None)
        price_range = getattr(trade, 'price_range', None)
        
        if max_rr is None:
            raise ValueError("Trade tuple does not have 'max_rr' attribute")
        if price_range is None:
            raise ValueError("Trade tuple does not have 'price_range' attribute")
        
        # Get timestamps if available
        if has_timestamps:
            entry_time = getattr(trade, 'entry_time', None)
            exit_time = getattr(trade, 'exit_time', None)
        else:
            # Use trade number as proxy for time
            entry_time = datetime.now() + timedelta(minutes=idx)
            exit_time = entry_time + timedelta(minutes=5)  # Assume 5-minute trades
        
        # Calculate risk amount for this trade
        risk_amount = current_balance * risk_per_trade
        
        # Calculate position size using actual price_range (stop loss distance)
        # Convert price_range to pips (assuming 4 decimal places for most forex pairs)
        stop_distance_pips = price_range * 10000  # Convert to pips
        pip_value = 10.0  # $10 per pip for standard lot (100,000 units)
        position_size_lots = risk_amount / (stop_distance_pips * pip_value)
        
        # Calculate commission (round-turn: open + close)
        # Commission per lot is $7 round-turn, so multiply by number of lots
        commission_per_side = position_size_lots * (commission_per_lot / 2)  # $3.50 per lot per side
        commission_open = commission_per_side  # Charged when opening
        commission_close = commission_per_side  # Charged when closing
        total_commission = commission_open + commission_close  # $7 per lot total
        
        # Calculate trade P&L based on current balance
        if max_rr == -1:
            # Losing trade - lose the risk amount
            trade_pnl = -risk_amount
        else:
            # Winning trade - win RR * risk amount
            trade_pnl = max_rr * risk_amount
        
        # Calculate slippage (random cost per trade)
        # Typical slippage: 0.5-2 pips per trade
        slippage_pips = random.uniform(0.5, 2.0)  # Random between 0.5-2 pips
        slippage_cost = position_size_lots * slippage_pips * pip_value
        
        # Net P&L after commission AND slippage
        net_pnl = trade_pnl - total_commission - slippage_cost
        
        # Update balance
        current_balance += net_pnl
        
        # Track position opening and closing
        position_risk = current_balance * risk_per_trade
        
        # Add new position
        open_positions.append({
            'entry_time': entry_time,
            'exit_time': exit_time,
            'risk': position_risk
        })
        
        # Remove closed positions
        open_positions = [pos for pos in open_positions if pos['exit_time'] > entry_time]
        
        # Calculate current exposure
        current_exposure = sum(pos['risk'] for pos in open_positions)
        max_simultaneous_positions = max(max_simultaneous_positions, len(open_positions))
        
        # Track daily exposure
        if has_timestamps and entry_time is not None:
            date_key = entry_time.date()
            if date_key not in daily_exposure:
                daily_exposure[date_key] = []
            daily_exposure[date_key].append(current_exposure)
        # Store equity data
        equity_curve.append({
            'trade_number': idx,
            'max_rr': max_rr,
            'trade_pnl': trade_pnl,
            'commission': total_commission,
            'slippage': slippage_cost,
            'net_pnl': net_pnl,
            'balance': current_balance,
            'cumulative_return': (current_balance - starting_balance) / starting_balance * 100,
            'open_positions': len(open_positions),
            'current_exposure': current_exposure,
            'exposure_pct': (current_exposure / current_balance) * 100 if current_balance > 0 else 0,
            'position_size_lots': position_size_lots,
            'stop_distance_pips': stop_distance_pips
        })
    
    # Convert to DataFrame
    equity_df = pd.DataFrame(equity_curve)
    
    # Calculate additional metrics
    total_trades = len(equity_df)
    winning_trades = (equity_df['max_rr'] > 0).sum()
    losing_trades = (equity_df['max_rr'] == -1).sum()
    win_rate = winning_trades / total_trades * 100
    
    final_balance = equity_df['balance'].iloc[-1]
    total_return = (final_balance - starting_balance) / starting_balance * 100
    
    # Calculate drawdown
    equity_df['peak'] = equity_df['balance'].expanding().max()
    equity_df['drawdown'] = (equity_df['balance'] - equity_df['peak']) / equity_df['peak'] * 100
    max_drawdown = equity_df['drawdown'].min()
    
    # Calculate exposure statistics
    avg_exposure = equity_df['exposure_pct'].mean()
    max_exposure = equity_df['exposure_pct'].max()
    
    # Add debugging for exposure calculation
    print(f"\nExposure Debug Info:")
    print(f"Exposure column sample values: {equity_df['exposure_pct'].head(10).tolist()}")
    print(f"Exposure column stats:")
    print(f"  Min: {equity_df['exposure_pct'].min():.4f}%")
    print(f"  Max: {equity_df['exposure_pct'].max():.4f}%")
    print(f"  Mean: {equity_df['exposure_pct'].mean():.4f}%")
    print(f"  Median: {equity_df['exposure_pct'].median():.4f}%")
    print(f"  Std Dev: {equity_df['exposure_pct'].std():.4f}%")
    
    # Check for any NaN or infinite values
    print(f"  NaN values: {equity_df['exposure_pct'].isna().sum()}")
    print(f"  Infinite values: {np.isinf(equity_df['exposure_pct']).sum()}")
    
    print(f"\nEquity Curve Summary:")
    print(f"Final balance: ${final_balance:,.2f}")
    print(f"Total return: {total_return:.2f}%")
    print(f"Total trades: {total_trades}")
    print(f"Winning trades: {winning_trades} ({win_rate:.1f}%)")
    print(f"Losing trades: {losing_trades}")
    print(f"Max drawdown: {max_drawdown:.2f}%")
    print(f"\nPosition Management:")
    print(f"Max simultaneous positions: {max_simultaneous_positions}")
    print(f"Average exposure: {avg_exposure:.2f}%")
    print(f"Max exposure: {max_exposure:.2f}%")
    
    return equity_df

def plot_equity_curve(equity_df: pd.DataFrame, save_path: str = ""):
    """
    Plot equity curve with key metrics including position tracking
    """
    # Set dark theme
    plt.style.use('dark_background')
    
    # Create equity subfolder
    if save_path:
        base_dir = os.path.dirname(save_path)
        equity_dir = os.path.join(base_dir, "equity")
        os.makedirs(equity_dir, exist_ok=True)
        base_name = os.path.splitext(os.path.basename(save_path))[0]
    else:
        equity_dir = ""
        base_name = "equity_curve"
    
    # Plot 1: Equity Curve
    fig1, ax1 = plt.subplots(1, 1, figsize=(12, 8))
    ax1.plot(equity_df['trade_number'], equity_df['balance'], 'lime', linewidth=1.5, label='Account Balance')
    ax1.axhline(y=equity_df['balance'].iloc[0], color='red', linestyle='--', alpha=0.7, label='Starting Balance')
    ax1.set_title('Equity Curve Over Time', fontsize=14, fontweight='bold', color='white')
    ax1.set_xlabel('Trade Number', color='white')
    ax1.set_ylabel('Account Balance ($)', color='white')
    ax1.legend()
    ax1.grid(True, alpha=0.3, color='gray')
    ax1.tick_params(colors='white')
    ax1.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'${x:,.0f}'))
    
    if save_path:
        plot1_path = os.path.join(equity_dir, f"{base_name}_equity_curve.png")
        plt.savefig(plot1_path, dpi=300, bbox_inches='tight', facecolor='black', edgecolor='none')
        print(f"Equity curve plot saved to: {plot1_path}")
    plt.close(fig1)
    
    # Plot 2: Drawdown
    fig2, ax2 = plt.subplots(1, 1, figsize=(12, 8))
    ax2.fill_between(equity_df['trade_number'], equity_df['drawdown'], 0, color='red', alpha=0.3)
    ax2.plot(equity_df['trade_number'], equity_df['drawdown'], 'red', linewidth=1.5)
    ax2.set_title('Drawdown (%)', fontsize=14, fontweight='bold', color='white')
    ax2.set_xlabel('Trade Number', color='white')
    ax2.set_ylabel('Drawdown (%)', color='white')
    ax2.grid(True, alpha=0.3, color='gray')
    ax2.tick_params(colors='white')
    
    if save_path:
        plot2_path = os.path.join(equity_dir, f"{base_name}_drawdown.png")
        plt.savefig(plot2_path, dpi=300, bbox_inches='tight', facecolor='black', edgecolor='none')
        print(f"Drawdown plot saved to: {plot2_path}")
    plt.close(fig2)
    
    # Plot 3: Risk Exposure
    fig3, ax3 = plt.subplots(1, 1, figsize=(12, 8))
    ax3.plot(equity_df['trade_number'], equity_df['exposure_pct'], 'yellow', linewidth=1.5, label='Risk Exposure %')
    ax3.axhline(y=5, color='orange', linestyle='--', alpha=0.7, label='5% Exposure Limit')
    ax3.axhline(y=10, color='red', linestyle='--', alpha=0.7, label='10% Exposure Limit')
    ax3.set_title('Risk Exposure Over Time', fontsize=14, fontweight='bold', color='white')
    ax3.set_xlabel('Trade Number', color='white')
    ax3.set_ylabel('Risk Exposure (%)', color='white')
    ax3.legend()
    ax3.grid(True, alpha=0.3, color='gray')
    ax3.tick_params(colors='white')
    
    if save_path:
        plot3_path = os.path.join(equity_dir, f"{base_name}_risk_exposure.png")
        plt.savefig(plot3_path, dpi=300, bbox_inches='tight', facecolor='black', edgecolor='none')
        print(f"Risk exposure plot saved to: {plot3_path}")
    plt.close(fig3)
    
    # Plot 4: Simultaneous Open Positions
    fig4, ax4 = plt.subplots(1, 1, figsize=(12, 8))
    ax4.plot(equity_df['trade_number'], equity_df['open_positions'], 'cyan', linewidth=1.5, label='Open Positions')
    ax4.axhline(y=5, color='orange', linestyle='--', alpha=0.7, label='5 Position Limit')
    ax4.axhline(y=10, color='red', linestyle='--', alpha=0.7, label='10 Position Limit')
    ax4.set_title('Simultaneous Open Positions', fontsize=14, fontweight='bold', color='white')
    ax4.set_xlabel('Trade Number', color='white')
    ax4.set_ylabel('Number of Open Positions', color='white')
    ax4.legend()
    ax4.grid(True, alpha=0.3, color='gray')
    ax4.tick_params(colors='white')
    
    if save_path:
        plot4_path = os.path.join(equity_dir, f"{base_name}_open_positions.png")
        plt.savefig(plot4_path, dpi=300, bbox_inches='tight', facecolor='black', edgecolor='none')
        print(f"Open positions plot saved to: {plot4_path}")
    plt.close(fig4)

def save_equity_data(equity_df: pd.DataFrame, save_path: str):
    """
    Save equity curve data to CSV
    
    Args:
        equity_df: DataFrame with equity curve data
        save_path: Path to save the CSV file
    """
    # Create equity subfolder
    base_dir = os.path.dirname(save_path)
    equity_dir = os.path.join(base_dir, "equity")
    os.makedirs(equity_dir, exist_ok=True)
    
    # Save to equity subfolder
    base_name = os.path.splitext(os.path.basename(save_path))[0]
    equity_csv_path = os.path.join(equity_dir, f"{base_name}_equity_data.csv")
    
    equity_df.to_csv(equity_csv_path, index=False, float_format='%.6f')
    print(f"Equity curve data saved to: {equity_csv_path}")

def main():
    """Main function to run equity curve analysis"""
    if len(sys.argv) < 2:
        print("Usage: python equity_plotter.py <csv_filepath> [starting_balance] [risk_per_trade] [commission_per_trade]")
        print("Example: python equity_plotter.py analysis/EUR_USD_20250719_224249_filtered/EUR_USD_2014_2023_fixed.csv")
        print("Example: python equity_plotter.py analysis/EUR_USD_20250719_224249_filtered/EUR_USD_2014_2023_fixed.csv 100000 0.01 0.0001")
        return
    
    csv_path = sys.argv[1]
    starting_balance = float(sys.argv[2]) if len(sys.argv) > 2 else 100000
    risk_per_trade = float(sys.argv[3]) if len(sys.argv) > 3 else 0.01
    commission_per_trade = float(sys.argv[4]) if len(sys.argv) > 4 else 0.0001
    
    if not os.path.exists(csv_path):
        print(f"Error: CSV file not found: {csv_path}")
        return
    
    try:
        # Calculate equity curve
        equity_df = calculate_equity_curve(csv_path, starting_balance, risk_per_trade, commission_per_trade)
        
        # Create output paths
        base_dir = os.path.dirname(csv_path)
        base_name = os.path.splitext(os.path.basename(csv_path))[0]
        
        plot_path = os.path.join(base_dir, f"{base_name}_equity_curve.png")
        data_path = os.path.join(base_dir, f"{base_name}_equity_data.csv")
        
        # Plot equity curve
        plot_equity_curve(equity_df, plot_path)
        
        # Save equity data
        save_equity_data(equity_df, data_path)
        
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main() 