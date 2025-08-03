import json
import os
import shutil
import subprocess
import sys
import pandas as pd
import numpy as np
from datetime import datetime
import glob

class MultiAssetTester:
    def __init__(self, parameters_path: str = "utils/parameters.json"):
        self.parameters_path = parameters_path
        self.analysis_base_dir = "analysis/group_backtests"
        self.results = {}  # Store results for each asset
        
    def load_parameters(self):
        """Load current parameters"""
        with open(self.parameters_path, 'r') as f:
            return json.load(f)
    
    def save_parameters(self, params):
        """Save parameters back to file"""
        with open(self.parameters_path, 'w') as f:
            json.dump(params, f, indent=2)
    
    def get_parquet_files(self):
        """Get all parquet files from the data directory"""
        params = self.load_parameters()
        parquet_path = params['parquet_path']
        
        if not os.path.exists(parquet_path):
            print(f"Error: Parquet path does not exist: {parquet_path}")
            return []
        
        parquet_files = glob.glob(os.path.join(parquet_path, "*.parquet"))
        return sorted(parquet_files)
    
    def extract_asset_name(self, parquet_path):
        """Extract asset name from parquet filename"""
        filename = os.path.basename(parquet_path)
        # Remove '_compiled.parquet' suffix
        asset_name = filename.replace('_compiled.parquet', '')
        return asset_name
    
    def update_parameters_for_asset(self, asset_name):
        """Update parameters for a specific asset"""
        params = self.load_parameters()
        
        # Extract base and quote from asset name (e.g., "EUR_USD" -> base="EUR", quote="USD")
        if '_' in asset_name:
            base, quote = asset_name.split('_', 1)
            params['base'] = base
            params['quote'] = quote
            params['use_specific_asset'] = True
        else:
            print(f"Warning: Could not parse asset name: {asset_name}")
            return False
        
        self.save_parameters(params)
        return True
    
    def run_backtest(self, asset_name):
        """Run a single backtest for an asset"""
        try:
            print(f"\nRunning backtest for {asset_name}...")
            
            # Update parameters for this asset
            if not self.update_parameters_for_asset(asset_name):
                return None
            
            # Run the backtest
            result = subprocess.run([sys.executable, "run_single_test.py"], 
                                  capture_output=True, text=True, timeout=600)
            
            if result.returncode != 0:
                print(f"Backtest failed for {asset_name}: {result.stderr}")
                return None
            
            print(f"Backtest completed successfully for {asset_name}")
            return True
            
        except subprocess.TimeoutExpired:
            print(f"Backtest timed out for {asset_name}")
            return None
        except Exception as e:
            print(f"Error running backtest for {asset_name}: {str(e)}")
            return None
    
    def copy_backtest_results(self, asset_name):
        """Copy backtest results to analysis folder"""
        try:
            params = self.load_parameters()
            output_dir = params['output_dir']
            output_folders = [f for f in os.listdir(output_dir) if os.path.isdir(os.path.join(output_dir, f))]
            
            if not output_folders:
                print(f"No output folders found for {asset_name}")
                return None
            
            latest_folder = max(output_folders, key=lambda x: os.path.getctime(os.path.join(output_dir, x)))
            source_path = os.path.join(output_dir, latest_folder)
            
            # Create analysis folder name
            constraint_value = params.get('const_norm_price_range_upper', 'unknown')
            year = params['start_date'][:4]
            analysis_folder = f"multiasset_{constraint_value}_{year}"
            
            analysis_path = os.path.join(self.analysis_base_dir, analysis_folder)
            os.makedirs(analysis_path, exist_ok=True)
            
            dest_path = os.path.join(analysis_path, asset_name)
            shutil.copytree(source_path, dest_path)
            
            #print(f"Results copied to: {dest_path}")
            return dest_path
            
        except Exception as e:
            print(f"Error copying results for {asset_name}: {str(e)}")
            return None
    
    def process_csv(self, backtest_path):
        """Run fix_and_sample_csv on the backtest results"""
        try:
            csv_files = [f for f in os.listdir(backtest_path) if f.endswith('.csv')]
            if not csv_files:
                print(f"No CSV files found in {backtest_path}")
                return None
            
            csv_path = os.path.join(backtest_path, csv_files[0])
            
            result = subprocess.run([
                sys.executable, "analysis/fix_and_sample_csv.py", csv_path
            ], capture_output=True, text=True, timeout=60)
            
            if result.returncode != 0:
                print(f"CSV processing failed: {result.stderr}")
                return None
            
            fixed_csv_path = csv_path.replace('.csv', '_fixed.csv')
            if os.path.exists(fixed_csv_path):
                return fixed_csv_path
            else:
                print(f"Fixed CSV not found: {fixed_csv_path}")
                return None
                
        except Exception as e:
            print(f"Error processing CSV: {str(e)}")
            return None
    
    def calculate_expectancy(self, csv_path, asset_name):
        """Calculate expectancy for an asset"""
        try:
            result = subprocess.run([
                sys.executable, "analysis/expectancy_calculator.py", csv_path
            ], capture_output=True, text=True, timeout=60)
            
            if result.returncode != 0:
                print(f"Expectancy calculation failed for {asset_name}: {result.stderr}")
                return None
            
            # Parse the output to extract expectancies
            output_lines = result.stdout.split('\n')
            expectancies = {}
            
            # Look for the summary table format
            for line in output_lines:
                parts = line.strip().split()
                if len(parts) >= 6:
                    try:
                        # Check if first part looks like a RR value (1.0, 1.5, 2.0, etc.)
                        rr_str = parts[0]
                        if rr_str.replace('.', '').replace('0', '').replace('1', '').replace('2', '').replace('3', '').replace('4', '').replace('5', '').replace('6', '').replace('7', '').replace('8', '').replace('9', '') == '':
                            rr = float(rr_str)
                            # Format: RR Hits Stops Total Winrate Expectancy
                            # Index:  0    1     2     3      4         5
                            total_trades = int(parts[3])  # Total trades column (index 3)
                            expectancy = float(parts[5])  # Expectancy column (index 5)
                            expectancies[rr] = {
                                'total_trades': total_trades,
                                'expectancy': expectancy
                            }
                            print(f"  Parsed {rr}RR: {total_trades} trades, expectancy {expectancy:.6f}")
                    except (ValueError, IndexError) as e:
                        continue
            
            if not expectancies:
                print(f"Warning: No expectancy data parsed for {asset_name}")
                print(f"Raw output: {result.stdout}")
            
            return expectancies
            
        except Exception as e:
            print(f"Error calculating expectancy for {asset_name}: {str(e)}")
            return None
    
    def copy_fixed_csvs(self, analysis_folder):
        """Copy all fixed CSVs to a dedicated folder"""
        try:
            fixed_csvs_dir = os.path.join(analysis_folder, "fixed_csvs")
            os.makedirs(fixed_csvs_dir, exist_ok=True)
            
            # Find all fixed CSV files in the analysis folder
            fixed_csv_pattern = os.path.join(analysis_folder, "**", "*_fixed.csv")
            fixed_csv_files = glob.glob(fixed_csv_pattern, recursive=True)
            
            for csv_file in fixed_csv_files:
                asset_name = os.path.basename(os.path.dirname(csv_file))
                dest_file = os.path.join(fixed_csvs_dir, f"{asset_name}_fixed.csv")
                shutil.copy2(csv_file, dest_file)
                print(f"Copied fixed CSV: {dest_file}")
            
            return fixed_csvs_dir
            
        except Exception as e:
            print(f"Error copying fixed CSVs: {str(e)}")
            return None
    
    def save_results_csv(self, analysis_folder):
        """Save results to CSV file"""
        try:
            results_data = []
            
            for asset_name, asset_results in self.results.items():
                row = {'Asset': asset_name}
                
                # Add trade counts for each RR level (grouped together)
                for rr in [1.0, 1.5, 2.0, 2.5, 3.0]:
                    if rr in asset_results:
                        row[f'{rr}rr_trades'] = asset_results[rr]['total_trades']
                    else:
                        row[f'{rr}rr_trades'] = 0
                
                # Add expectancies for each RR level (grouped together) - rounded to 5 decimal places
                for rr in [1.0, 1.5, 2.0, 2.5, 3.0]:
                    if rr in asset_results:
                        row[f'{rr}rr_expectancy'] = round(asset_results[rr]['expectancy'], 5)
                    else:
                        row[f'{rr}rr_expectancy'] = 0.0
                
                # Add total RR generated for each RR level (expectancy * trades) - rounded to 5 decimal places
                for rr in [1.0, 1.5, 2.0, 2.5, 3.0]:
                    if rr in asset_results:
                        total_rr = asset_results[rr]['expectancy'] * asset_results[rr]['total_trades']
                        row[f'{rr}rr_total_rr'] = round(total_rr, 5)
                    else:
                        row[f'{rr}rr_total_rr'] = 0.0
                
                results_data.append(row)
            
            # Create DataFrame and save
            df = pd.DataFrame(results_data)
            results_path = os.path.join(analysis_folder, "multiasset_results.csv")
            df.to_csv(results_path, index=False)
            print(f"Results saved to: {results_path}")
            
            return results_path
            
        except Exception as e:
            print(f"Error saving results CSV: {str(e)}")
            return None
    
    def cleanup_output_folder(self):
        """Clean up the output folder after all tests are complete"""
        try:
            params = self.load_parameters()
            output_dir = params['output_dir']
            
            if not os.path.exists(output_dir):
                print("Output directory does not exist, nothing to clean up")
                return
            
            output_folders = [f for f in os.listdir(output_dir) if os.path.isdir(os.path.join(output_dir, f))]
            
            if not output_folders:
                print("No output folders found to clean up")
                return
            
            print(f"\nCleaning up output folder: {output_dir}")
            print(f"Found {len(output_folders)} folders to remove")
            
            for folder in output_folders:
                folder_path = os.path.join(output_dir, folder)
                try:
                    shutil.rmtree(folder_path)
                    print(f"Removed: {folder}")
                except Exception as e:
                    print(f"Error removing {folder}: {str(e)}")
            
            print("Output folder cleanup complete")
            
        except Exception as e:
            print(f"Error during cleanup: {str(e)}")
    
    def run_multi_asset_test(self):
        """Run backtests for all assets"""
        parquet_files = self.get_parquet_files()
        
        if not parquet_files:
            print("No parquet files found")
            return
        
        print(f"Found {len(parquet_files)} parquet files to test")
        
        # Create analysis folder name
        params = self.load_parameters()
        constraint_value = params.get('const_norm_price_range_upper', 'unknown')
        year = params['start_date'][:4]
        analysis_folder = f"multiasset_{constraint_value}_{year}"
        analysis_path = os.path.join(self.analysis_base_dir, analysis_folder)
        
        print(f"Results will be saved to: {analysis_path}")
        
        # Test each asset
        for parquet_file in parquet_files:
            asset_name = self.extract_asset_name(parquet_file)
            print(f"\n{'='*60}")
            print(f"Testing asset: {asset_name}")
            print(f"Parquet file: {parquet_file}")
            print(f"{'='*60}")
            
            # Run backtest
            if not self.run_backtest(asset_name):
                print(f"Skipping {asset_name} due to backtest failure")
                continue
            
            # Copy results
            backtest_path = self.copy_backtest_results(asset_name)
            if not backtest_path:
                print(f"Skipping {asset_name} due to copy failure")
                continue
            
            # Process CSV
            csv_path = self.process_csv(backtest_path)
            if not csv_path:
                print(f"Skipping {asset_name} due to CSV processing failure")
                continue
            
            # Calculate expectancy
            expectancies = self.calculate_expectancy(csv_path, asset_name)
            if expectancies is None:
                print(f"Skipping {asset_name} due to expectancy calculation failure")
                continue
            
            # Store results
            self.results[asset_name] = expectancies
        
        # Copy fixed CSVs to dedicated folder
        self.copy_fixed_csvs(analysis_path)
        
        # Save results to CSV
        self.save_results_csv(analysis_path)
        
        # Clean up output folder
        self.cleanup_output_folder()
        
        # Print summary
        print(f"\n{'='*80}")
        print(f"MULTI-ASSET TESTING SUMMARY")
        print(f"{'='*80}")
        print(f"Total assets tested: {len(self.results)}")
        print(f"Results saved to: {analysis_path}")
        
        if self.results:
            print(f"\nAsset Results Summary:")
            for asset_name, asset_results in self.results.items():
                print(f"\n{asset_name}:")
                for rr in [1.0, 1.5, 2.0, 2.5, 3.0]:
                    if rr in asset_results:
                        trades = asset_results[rr]['total_trades']
                        expectancy = asset_results[rr]['expectancy']
                        print(f"  {rr}RR: {trades} trades, expectancy {expectancy:.6f}")
        
        return self.results

def main():
    """Main function to run multi-asset testing"""
    print("Starting multi-asset testing...")
    
    tester = MultiAssetTester()
    results = tester.run_multi_asset_test()
    
    print(f"\nMulti-asset testing complete!")

if __name__ == "__main__":
    main() 