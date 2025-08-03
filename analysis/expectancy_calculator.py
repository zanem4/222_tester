import pandas as pd
import numpy as np
import json
import sys
import os

def load_tp_multipliers(parameters_path: str = "utils/parameters.json"):
    """
    Load TP multipliers from parameters.json file
    
    Args:
        parameters_path: Path to parameters.json file
    
    Returns:
        List of TP multipliers
    """
    try:
        with open(parameters_path, 'r') as f:
            params = json.load(f)
        return params.get('tp_multipliers', [])
    except Exception as e:
        print(f"Error: Could not load parameters.json: {e}")
        return []

def calculate_expectancy(csv_path: str, tp_multipliers: list):
    """
    Calculate expectancy for each RR model
    
    Args:
        csv_path: Path to the CSV file with trade results
        tp_multipliers: List of TP multipliers (RR levels)
    
    Returns:
        Dictionary with results for each model
    """
    print(f"Loading CSV: {csv_path}")
    df = pd.read_csv(csv_path)
    
    if 'max_rr' not in df.columns:
        raise ValueError("CSV must contain 'max_rr' column")
    
    print(f"Total trades: {len(df)}")
    print(f"TP multipliers: {tp_multipliers}")
    
    # Debug: Show distribution of max_rr values
    print(f"\nMax RR distribution:")
    rr_counts = df['max_rr'].value_counts().sort_index()
    for rr, count in rr_counts.items():
        print(f"  {rr}: {count} trades")
    
    results = {}
    
    for rr in tp_multipliers:
        print(f"\n--- Analyzing {rr}RR model ---")
        
        # For each RR level, we want to count:
        # 1. Trades that hit this RR level or higher (max_rr >= rr)
        # 2. Trades that hit stop loss (max_rr == -1)
        
        # Count hits: trades where max_rr >= this RR level
        hits = (df['max_rr'] >= rr).sum()
        
        # Count stops: trades that hit stop loss
        stops = (df['max_rr'] == -1).sum()
        
        # Total trades for this model = ALL trades in the dataset
        # This is because every trade is relevant to every RR model
        total_trades = len(df)
        
        if total_trades == 0:
            print(f"  No trades for {rr}RR model")
            results[rr] = {
                'hits': 0,
                'stops': 0,
                'total_trades': 0,
                'winrate': 0.0,
                'expectancy': 0.0
            }
            continue
        
        # Calculate winrate
        winrate = hits / total_trades
        
        # Calculate expectancy: EV = WR*RR - (1-WR)
        expectancy = (winrate * rr) - (1 - winrate)
        
        print(f"  Hits: {hits}")
        print(f"  Stops: {stops}")
        print(f"  Total trades: {total_trades}")
        print(f"  Winrate: {winrate:.4f} ({winrate*100:.2f}%)")
        print(f"  Expectancy: {expectancy:.6f}")
        
        results[rr] = {
            'hits': hits,
            'stops': stops,
            'total_trades': total_trades,
            'winrate': winrate,
            'expectancy': expectancy
        }
    
    return results

def print_summary(results: dict):
    """Print a summary table of all results"""
    print(f"\n{'='*60}")
    print(f"EXPECTANCY ANALYSIS SUMMARY")
    print(f"{'='*60}")
    print(f"{'RR':>6} {'Hits':>8} {'Stops':>8} {'Total':>8} {'Winrate':>10} {'Expectancy':>12}")
    print(f"{'-'*60}")
    
    for rr in sorted(results.keys()):
        r = results[rr]
        print(f"{rr:>6.1f} {r['hits']:>8} {r['stops']:>8} {r['total_trades']:>8} "
              f"{r['winrate']*100:>9.2f}% {r['expectancy']:>11.6f}")

def main():
    """Main function to run expectancy calculation"""
    if len(sys.argv) < 2:
        print("Usage: python expectancy_calculator.py <csv_filepath>")
        print("Example: python expectancy_calculator.py analysis/EUR_USD_20250719_224249_filtered/EUR_USD_2014_2023_fixed.csv")
        return
    
    csv_path = sys.argv[1]
    
    if not os.path.exists(csv_path):
        print(f"Error: CSV file not found: {csv_path}")
        return
    
    try:
        # Load TP multipliers from parameters
        tp_multipliers = load_tp_multipliers()
        
        if not tp_multipliers:
            print("Error: No TP multipliers found in parameters.json")
            return
        
        # Calculate expectancy
        results = calculate_expectancy(csv_path, tp_multipliers)
        
        # Print summary
        print_summary(results)
        
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main() 