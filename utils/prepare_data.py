import pyarrow.parquet as pq
import numpy as np
from datetime import datetime

def prepare_data(file_path, start_date=None, stop_date=None):
    """
    Read and filter parquet data based on date range
    
    Args:
        file_path: Path to the parquet file
        start_date: Start date as ISO datetime string (e.g., "2024-01-01T00:00:00")
        stop_date: Stop date as ISO datetime string (e.g., "2024-12-31T23:59:59")
    
    Returns:
        tuple: (time, open, high, low, close, volume) arrays
    """
    # Read the selected parquet file
    table = pq.read_table(file_path)
    times: np.ndarray = table["Time"].to_numpy()
    
    # Create mask for date filtering
    mask = np.ones(len(times), dtype=bool)
    
    # Apply start date filter if provided
    if start_date:
        try:
            start_timestamp = datetime.fromisoformat(start_date).timestamp()
            mask &= (times >= start_timestamp)
        except ValueError as e:
            print(f"Warning: Invalid start_date format '{start_date}'. Error: {e}")
            print("Using default start date filter (times < 1718496000)")
            mask &= (times < 1718496000)
    
    # Apply stop date filter if provided
    if stop_date:
        try:
            stop_timestamp = datetime.fromisoformat(stop_date).timestamp()
            mask &= (times <= stop_timestamp)
        except ValueError as e:
            print(f"Warning: Invalid stop_date format '{stop_date}'. Error: {e}")
            print("Using default stop date filter (times < 1718496000)")
            mask &= (times < 1718496000)
    
    # If no date filters provided, use default filter
    if not start_date and not stop_date:
        mask = times < 1718496000
    
    # Apply mask to get filtered data
    time = times[mask]
    open_data = table["Open"].to_numpy()[mask]
    high = table["High"].to_numpy()[mask]
    low = table["Low"].to_numpy()[mask]
    close = table["Close"].to_numpy()[mask]
    volume = table["Volume"].to_numpy()[mask]
    
    # Print date range info
    if len(time) > 0:
        start_dt = datetime.fromtimestamp(time[0])
        end_dt = datetime.fromtimestamp(time[-1])
        print(f"Data range: {start_dt} to {end_dt}")
        print(f"Total bars in range: {len(time)}")
    else:
        print("Warning: No data found in specified date range")

    return time, open_data, high, low, close, volume