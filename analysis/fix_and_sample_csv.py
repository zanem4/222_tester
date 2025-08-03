import pandas as pd
import numpy as np
import sys
import os
import glob

def find_csv_files(analysis_dir: str = "analysis"):
    """
    Find all CSV files in analysis directory and subdirectories
    
    Args:
        analysis_dir: Path to analysis directory (default "analysis")
    
    Returns:
        List of CSV file paths
    """
    # Search for all CSV files recursively
    csv_pattern = os.path.join(analysis_dir, "**", "*.csv")
    csv_files = glob.glob(csv_pattern, recursive=True)
    
    # Filter out sample files (files with 'sample' in name)
    csv_files = [f for f in csv_files if 'sample' not in f.lower()]
    
    return csv_files

def fix_and_sample_csv(csv_path: str, sample_size: int = 5000, output_name: str = ""):
    """
    Fix NaN values in CSV, remove placeholder columns, and create a random sample for viewing
    
    Args:
        csv_path: Path to the original CSV file
        sample_size: Number of rows to randomly sample (default 5000)
        output_name: Custom output filename (optional)
    """
    try:
        print(f"Reading CSV file: {csv_path}")
        
        # Get file size
        file_size = os.path.getsize(csv_path) / (1024 * 1024)  # MB
        print(f"File size: {file_size:.2f} MB")
        
        # Read the CSV file
        df = pd.read_csv(csv_path)
        
        print(f"Original shape: {df.shape}")
        print(f"Original columns: {list(df.columns)}")
        
        # Remove placeholder columns
        placeholder_cols = ['placeholder1', 'placeholder2', 'placeholder3', 'placeholder4', 'placeholder5']
        cols_to_remove = [col for col in placeholder_cols if col in df.columns]
        
        if cols_to_remove:
            print(f"Removing placeholder columns: {cols_to_remove}")
            df = df.drop(columns=cols_to_remove)
            print(f"Shape after removing placeholders: {df.shape}")
        else:
            print("No placeholder columns found to remove")
        
        print(f"Columns with NaN values:")
        for col in df.columns:
            if df[col].isna().sum() > 0:
                print(f"  - {col}: {df[col].isna().sum()} NaN values")
        
        # Fix NaN values
        print("\nFixing NaN values...")
        for col in df.columns:
            if df[col].isna().sum() > 0:
                if 'timing' in col.lower():
                    # For timing columns, use 0.0
                    df[col] = df[col].fillna(0.0)
                elif df[col].dtype in ['float64', 'float32']:
                    # For other float columns, use 0.0
                    df[col] = df[col].fillna(0.0)
                else:
                    # For other columns, use empty string
                    df[col] = df[col].fillna('')
        
        # Fix max_rr for losing trades
        if 'max_rr' in df.columns and 'win_loss' in df.columns:
            print("\nFixing max_rr for losing trades...")
            losing_trades = df['win_loss'] == 0
            losing_count = losing_trades.sum()
            if losing_count > 0:
                print(f"Found {losing_count} losing trades with incorrect max_rr")
                # Set max_rr to -1 for losing trades where it's currently 0
                mask = (df['win_loss'] == 0) & (df['max_rr'] == 0)
                fixed_count = mask.sum()
                df.loc[mask, 'max_rr'] = -1.0
                print(f"Fixed {fixed_count} losing trades: max_rr set to -1")
            else:
                print("No losing trades found")
        
        # Save fixed CSV (without placeholders)
        fixed_path = csv_path.replace('.csv', '_fixed.csv')
        df.to_csv(fixed_path, index=False)
        print(f"Fixed CSV saved: {fixed_path}")
        
        # Create random sample
        if len(df) > sample_size:
            sample_df = df.sample(n=sample_size, random_state=42)
            print(f"Created random sample of {sample_size} rows")
        else:
            sample_df = df
            print(f"File has {len(df)} rows, using entire file as sample")
        
        # Save sample CSV
        if output_name:
            sample_path = os.path.join(os.path.dirname(csv_path), output_name)
        else:
            sample_path = csv_path.replace('.csv', '_sample.csv')
        
        sample_df.to_csv(sample_path, index=False)
        
        print(f"\nSample CSV saved: {sample_path}")
        print(f"Sample size: {len(sample_df)} rows")
        print(f"Sample file size: {os.path.getsize(sample_path) / 1024:.2f} KB")
        
        # Show sample statistics
        print(f"\nSample Statistics:")
        print(f"Columns: {len(sample_df.columns)}")
        print(f"Final columns: {list(sample_df.columns)}")
        print(f"Memory usage: {sample_df.memory_usage(deep=True).sum() / 1024:.2f} KB")
        
        return sample_path
        
    except Exception as e:
        print(f"Error processing {csv_path}: {str(e)}")
        return None

def process_all_csv_files(analysis_dir: str = "analysis", sample_size: int = 5000):
    """
    Process all CSV files found in analysis directory
    
    Args:
        analysis_dir: Path to analysis directory
        sample_size: Number of rows to sample
    """
    csv_files = find_csv_files(analysis_dir)
    
    if not csv_files:
        print(f"No CSV files found in {analysis_dir} directory")
        return
    
    print(f"Found {len(csv_files)} CSV file(s) to process:")
    for i, csv_file in enumerate(csv_files, 1):
        print(f"  {i}. {csv_file}")
    
    print(f"\nProcessing {len(csv_files)} file(s)...")
    
    processed_files = []
    for csv_file in csv_files:
        print(f"\n{'='*60}")
        print(f"Processing: {os.path.basename(csv_file)}")
        print(f"{'='*60}")
        
        sample_path = fix_and_sample_csv(csv_file, sample_size)
        if sample_path:
            processed_files.append(sample_path)
    
    print(f"\n{'='*60}")
    print(f"PROCESSING COMPLETE")
    print(f"{'='*60}")
    print(f"Processed {len(processed_files)} file(s)")
    print(f"Sample files created:")
    for sample_file in processed_files:
        print(f"  - {sample_file}")
    
    if processed_files:
        print(f"\nTo view with Excel Viewer extension, open:")
        print(f"  {processed_files[0]}")

if __name__ == "__main__":
    # Check if specific file path provided
    if len(sys.argv) > 1:
        csv_path = sys.argv[1]
        sample_size = int(sys.argv[2]) if len(sys.argv) > 2 else 5000
        fix_and_sample_csv(csv_path, sample_size)
    else:
        # Process all CSV files in analysis directory
        sample_size = int(sys.argv[1]) if len(sys.argv) > 1 else 5000
        process_all_csv_files("analysis", sample_size) 