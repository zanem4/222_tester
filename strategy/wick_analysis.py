import numpy as np

def calculate_wick_sizes(setup_idx: int, high: np.ndarray, low: np.ndarray, 
                        close: np.ndarray, open_array: np.ndarray) -> np.ndarray:
    """
    Calculate upwick and downwick sizes for all 3 bars in the 222 pattern
    
    Args:
        setup_idx: Index of the setup (bar 3 in the 222 pattern)
        high: High prices array
        low: Low prices array
        close: Close prices array
        open_array: Open prices array
    
    Returns:
        numpy array: [upwick1, downwick1, upwick2, downwick2, upwick3, downwick3]
    """
    bar1_idx = setup_idx - 2
    bar2_idx = setup_idx - 1
    bar3_idx = setup_idx
    
    # Calculate wick sizes for each bar
    # Upwick = high - max(close, open)
    # Downwick = min(close, open) - low
    
    # Bar 1
    upwick1 = high[bar1_idx] - max(close[bar1_idx], open_array[bar1_idx])
    downwick1 = min(close[bar1_idx], open_array[bar1_idx]) - low[bar1_idx]
    
    # Bar 2
    upwick2 = high[bar2_idx] - max(close[bar2_idx], open_array[bar2_idx])
    downwick2 = min(close[bar2_idx], open_array[bar2_idx]) - low[bar2_idx]
    
    # Bar 3
    upwick3 = high[bar3_idx] - max(close[bar3_idx], open_array[bar3_idx])
    downwick3 = min(close[bar3_idx], open_array[bar3_idx]) - low[bar3_idx]
    
    return np.array([upwick1, downwick1, upwick2, downwick2, upwick3, downwick3], dtype=np.float64) 