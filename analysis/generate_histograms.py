import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
from pathlib import Path

def find_fixed_csv_files(analysis_dir: str = "analysis"):
    """
    Find all fixed CSV files in analysis directory and subdirectories
    
    Args:
        analysis_dir: Path to analysis directory (default "analysis")
    
    Returns:
        List of fixed CSV file paths
    """
    # Search for all fixed CSV files recursively
    csv_pattern = os.path.join(analysis_dir, "**", "*_fixed.csv")
    csv_files = glob.glob(csv_pattern, recursive=True)
    
    return csv_files

def get_histogram_columns():
    """
    Get the list of columns to generate histograms for
    
    Returns:
        List of column names for histogram generation
    """
    return [
        # Timing columns
        'tp1_timing', 'tp2_timing', 'tp3_timing', 'tp4_timing', 'tp5_timing',
        
        # Price metrics
        'price_delta', 'sma', 'price_range', 'atr',
        
        # Body sizes
        'body1_size', 'body2_size', 'body3_size',
        
        # Wick columns
        'upwick1', 'downwick1', 'upwick2', 'downwick2', 'upwick3', 'downwick3',
        
        # Normalized metrics
        'norm_price_delta', 'norm_price_range', 'norm_body1', 'norm_body2', 'norm_body3',
        'norm_upwick1', 'norm_downwick1', 'norm_upwick2', 'norm_downwick2', 'norm_upwick3', 'norm_downwick3',
        
        # Drawdown
        'max_drawdown_pips'
    ]

def analyze_data_distribution(data: pd.Series, column_name: str):
    """
    Analyze the data distribution to determine optimal binning and outlier handling
    
    Args:
        data: Pandas Series containing the data
        column_name: Name of the column for analysis
    
    Returns:
        Dictionary with analysis results
    """
    clean_data = data.replace([np.inf, -np.inf], np.nan).dropna()
    
    if len(clean_data) == 0:
        return None
    
    analysis = {
        'total_count': len(clean_data),
        'zero_count': (clean_data == 0).sum(),
        'zero_percentage': (clean_data == 0).sum() / len(clean_data) * 100,
        'mean': clean_data.mean(),
        'std': clean_data.std(),
        'median': clean_data.median(),
        'q1': clean_data.quantile(0.25),
        'q3': clean_data.quantile(0.75),
        'iqr': clean_data.quantile(0.75) - clean_data.quantile(0.25),
        'min': clean_data.min(),
        'max': clean_data.max(),
        'p95': clean_data.quantile(0.95),
        'p99': clean_data.quantile(0.99)
    }
    
    # Determine if this is a wick column (likely to have many zeros)
    is_wick = any(wick in column_name.lower() for wick in ['upwick', 'downwick'])
    is_timing = 'timing' in column_name.lower()
    is_drawdown = 'drawdown' in column_name.lower()
    
    analysis['is_wick'] = is_wick
    analysis['is_timing'] = is_timing
    analysis['is_drawdown'] = is_drawdown
    
    return analysis

def get_optimal_bins_and_limits(data: pd.Series, column_name: str, analysis: dict):
    """
    Determine optimal bin count and data limits based on column type and distribution
    
    Args:
        data: Pandas Series containing the data
        column_name: Name of the column
        analysis: Analysis results from analyze_data_distribution
    
    Returns:
        Tuple of (bins, x_min, x_max, filtered_data)
    """
    if analysis is None:
        return 50, data.min(), data.max(), data
    
    # Initialize default values
    bins = 50
    x_min = data.min()
    x_max = data.max()
    
    # Determine optimal bin count
    if analysis['is_wick']:
        # For wick columns, use fewer bins due to many zeros
        bins = 30
    elif analysis['is_timing']:
        # For timing columns, use moderate bins
        bins = 40
    elif analysis['is_drawdown']:
        # For drawdown, use fewer bins
        bins = 35
    else:
        # For other columns, use more bins
        bins = 50
    
    # Determine data limits based on column type
    if analysis['is_wick']:
        # For wick columns, focus on non-zero values but keep some zeros
        non_zero_data = data[data > 0]
        if len(non_zero_data) > 0:
            x_min = 0  # Keep zeros
            x_max = pd.Series(non_zero_data).quantile(0.95)  # Focus on 95th percentile of non-zero
        else:
            x_min, x_max = data.min(), data.max()
    
    elif analysis['is_timing']:
        # For timing, focus on the main distribution
        x_min = data.quantile(0.01)
        x_max = data.quantile(0.95)
    
    elif analysis['is_drawdown']:
        # For drawdown, focus on positive values (negative is normal)
        positive_data = data[data > 0]
        if len(positive_data) > 0:
            x_min = data.min()  # Keep negative values
            x_max = pd.Series(positive_data).quantile(0.95)
        else:
            x_min, x_max = data.min(), data.max()
    
    else:
        # For other columns, use IQR-based limits
        q1, q3 = analysis['q1'], analysis['q3']
        iqr = analysis['iqr']
        x_min = max(data.min(), q1 - 1.5 * iqr)
        x_max = min(data.max(), q3 + 1.5 * iqr)
    
    # Ensure we have valid limits
    if x_min >= x_max:
        x_min = data.min()
        x_max = data.max()
    
    # Filter data to the determined range
    filtered_data = data[(data >= x_min) & (data <= x_max)]
    
    return bins, x_min, x_max, filtered_data

def create_histogram(data: pd.Series, column_name: str, output_path: str):
    """
    Create a relative frequency histogram for a single column
    
    Args:
        data: Pandas Series containing the data
        column_name: Name of the column (used for title)
        output_path: Path to save the histogram
    """
    # Filter out NaN and infinite values
    clean_data = data.replace([np.inf, -np.inf], np.nan).dropna()
    
    if len(clean_data) == 0:
        print(f"  Warning: No valid data for column '{column_name}'")
        return
    
    # Analyze the data distribution
    analysis = analyze_data_distribution(clean_data, column_name)
    
    # Check if analysis is valid before proceeding
    if analysis is not None:
        # Get optimal bins and limits
        bins, x_min, x_max, plot_data = get_optimal_bins_and_limits(clean_data, column_name, analysis)
        
        # Create the figure and axis
        plt.figure(figsize=(12, 8))
    else:
        print(f"  Error: Analysis for column '{column_name}' is None")
        return
    # Create histogram with relative frequency
    plt.hist(plot_data, bins=bins, density=True, alpha=0.7, color='skyblue', edgecolor='black')
    
    # Add title and labels
    plt.title(f'Relative Frequency Histogram: {column_name}', fontsize=16, fontweight='bold')
    plt.xlabel(column_name, fontsize=12)
    plt.ylabel('Relative Frequency', fontsize=12)
    
    # Add grid
    plt.grid(True, alpha=0.3)
    
    # Set x-axis limits
    plt.xlim(x_min, x_max)
    
    # Save the plot
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Print detailed analysis
    if analysis:
        print(f"  Created histogram: {os.path.basename(output_path)}")
        print(f"    Data points: {len(plot_data):,} (original: {len(clean_data):,})")
        print(f"    Bins: {bins}, Range: [{x_min:.6f}, {x_max:.6f}]")
        if analysis['is_wick']:
            print(f"    Zero values: {analysis['zero_count']:,} ({analysis['zero_percentage']:.1f}%)")
        print(f"    Mean: {analysis['mean']:.6f}, Std: {analysis['std']:.6f}, Median: {analysis['median']:.6f}")

def generate_histograms_for_csv(csv_path: str):
    """
    Generate histograms for all specified columns in a fixed CSV file
    
    Args:
        csv_path: Path to the fixed CSV file
    """
    try:
        print(f"Processing: {csv_path}")
        
        # Read the CSV file
        df = pd.read_csv(csv_path)
        print(f"  Loaded {len(df)} rows, {len(df.columns)} columns")
        
        # Get the directory containing the CSV
        csv_dir = os.path.dirname(csv_path)
        
        # Check if histograms folder already exists
        histograms_dir = os.path.join(csv_dir, "histograms")
        if os.path.exists(histograms_dir):
            print(f"  Histograms folder already exists, skipping: {histograms_dir}")
            return False
        
        # Create histograms directory
        os.makedirs(histograms_dir, exist_ok=True)
        print(f"  Created histograms directory: {histograms_dir}")
        
        # Get columns to process
        target_columns = get_histogram_columns()
        
        # Filter to only include columns that exist in the DataFrame
        available_columns = [col for col in target_columns if col in df.columns]
        missing_columns = [col for col in target_columns if col not in df.columns]
        
        if missing_columns:
            print(f"  Missing columns: {missing_columns}")
        
        print(f"  Generating histograms for {len(available_columns)} columns...")
        
        # Generate histogram for each column
        for column in available_columns:
            try:
                # Create output path
                output_filename = f"{column}_histogram.png"
                output_path = os.path.join(histograms_dir, output_filename)
                
                # Ensure the column data is a Series before creating histogram
                column_data = df[column]
                if isinstance(column_data, pd.Series):
                    create_histogram(column_data, column, output_path)
                else:
                    print(f"  Skipping {column}: Not a valid Series")
            except Exception as e:
                print(f"  Error creating histogram for {column}: {str(e)}")
        
        print(f"  Completed histograms for: {os.path.basename(csv_path)}")
        return True
        
    except Exception as e:
        print(f"Error processing {csv_path}: {str(e)}")
        return False

def process_all_fixed_csv_files(analysis_dir: str = "analysis"):
    """
    Process all fixed CSV files found in analysis directory
    
    Args:
        analysis_dir: Path to analysis directory
    """
    csv_files = find_fixed_csv_files(analysis_dir)
    
    if not csv_files:
        print(f"No fixed CSV files found in {analysis_dir} directory")
        return
    
    print(f"Found {len(csv_files)} fixed CSV file(s) to process:")
    for i, csv_file in enumerate(csv_files, 1):
        print(f"  {i}. {csv_file}")
    
    print(f"\nProcessing {len(csv_files)} file(s)...")
    
    processed_files = []
    for csv_file in csv_files:
        print(f"\n{'='*60}")
        success = generate_histograms_for_csv(csv_file)
        if success:
            processed_files.append(csv_file)
    
    print(f"\n{'='*60}")
    print(f"PROCESSING COMPLETE")
    print(f"{'='*60}")
    print(f"Processed {len(processed_files)} file(s)")
    print(f"Histogram folders created:")
    for csv_file in processed_files:
        histograms_dir = os.path.join(os.path.dirname(csv_file), "histograms")
        print(f"  - {histograms_dir}")

if __name__ == "__main__":
    # Process all fixed CSV files in analysis directory
    process_all_fixed_csv_files("analysis") 