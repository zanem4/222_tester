import numpy as np

def get_stop_level(setup_idx: int, high: np.ndarray, low: np.ndarray, is_long: bool) -> tuple:
    """
    Returns the stop level and the bar index it is defined from (bar1 in the 222 pattern).
    For long: stop = low of bar1. For short: stop = high of bar1.
    
    Args:
        setup_idx: Index of the setup (bar 3 in the 222 pattern)
        high: High prices array
        low: Low prices array
        is_long: True for long setup, False for short setup
    
    Returns:
        tuple: (stop_level, stop_bar_idx)
    """
    bar1_idx = setup_idx - 2  # First bar in 222 pattern
    if is_long:
        return float(low[bar1_idx]), bar1_idx
    else:
        return float(high[bar1_idx]), bar1_idx