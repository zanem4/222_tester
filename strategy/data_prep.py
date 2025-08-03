import numpy as np

def prepare_metrics_data(setup_time: np.float64, time_array: np.ndarray, high: np.ndarray, 
                        low: np.ndarray, close: np.ndarray, sma_period: int = 30) -> tuple:
    """
    Prepare data window for metrics calculation
    
    Args:
        setup_time: UNIX timestamp of setup (positive=long, negative=short)
        time_array: Array of UNIX timestamps (increments by 60)
        high: High prices array
        low: Low prices array
        close: Close prices array
        sma_period: Period for SMA calculation (default 30)
    
    Returns:
        tuple: (setup_idx, is_long, high_window, low_window, close_window) or (None, None, None, None, None) if invalid
    """
    # Convert timestamp to index using exact matching
    actual_time: np.float64 = abs(setup_time)
    setup_idx: np.int64 = np.where(time_array == actual_time)[0][0]
    
    # Determine setup direction
    is_long: np.bool_ = setup_time > 0
    
    # Ensure we have enough data and valid indices
    if setup_idx < sma_period - 1 or setup_idx >= len(high):
        return (None, None, None, None, None)
    
    # Get the window of data we need (sma_period bars ending at setup_idx)
    start_idx = max(0, int(setup_idx) - sma_period + 1)
    end_idx = int(setup_idx) + 1
    
    # Ensure we have enough data in the window
    if end_idx - start_idx < sma_period:
        return (None, None, None, None, None)
    
    high_window = high[start_idx:end_idx]
    low_window = low[start_idx:end_idx]
    close_window = close[start_idx:end_idx]
    
    return (setup_idx, is_long, high_window, low_window, close_window) 