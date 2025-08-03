import numpy as np

def calculate_price_delta(setup_idx: int, high: np.ndarray, low: np.ndarray, is_long: bool) -> np.float64:
    """
    Calculate price delta from 3-bar extrema for stop loss distance
    
    Args:
        setup_idx: Index of the setup (bar 3 in the 222 pattern)
        high: High prices array
        low: Low prices array
        is_long: True for long setup, False for short setup
    
    Returns:
        Price delta as float64
    """
    # Calculate price delta from 3-bar extrema
    bar1_idx = setup_idx - 2  # First bar in 222 pattern
    bar3_idx = setup_idx      # Third bar in 222 pattern (setup bar)
    
    if is_long:
        # High of bar3 - Low of bar1
        price_delta = high[bar3_idx] - low[bar1_idx]
    else:
        # High of bar1 - Low of bar3
        price_delta = high[bar1_idx] - low[bar3_idx]
    
    return np.float64(price_delta) 