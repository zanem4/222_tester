import numpy as np

def calculate_wick_size(high: np.float64, low: np.float64, close: np.float64, is_long: bool) -> np.float64:
    """
    Calculate wick size of the last candle in the 222 setup
    
    Args:
        high: High price of the last bar
        low: Low price of the last bar
        close: Close price of the last bar
        is_long: True for long setup, False for short setup
    
    Returns:
        Wick size as float64
    """
    if is_long:
        # For long setup: high - close (upper wick)
        wick_size = high - close
    else:
        # For short setup: close - low (lower wick)
        wick_size = close - low
    
    return np.float64(wick_size) 