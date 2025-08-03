import numpy as np
import numba

@numba.njit
def calculate_body_sizes(setup_idx: int, high: np.ndarray, low: np.ndarray, close: np.ndarray, open: np.ndarray) -> np.ndarray:
    """
    Calculate body sizes of the 3 bars in the 222 setup
    
    Args:
        setup_idx: Index of the setup bar (bar 3)
        high: High prices array
        low: Low prices array
        close: Close prices array
        open: Open prices array
    
    Returns:
        numpy array: [body1_size, body2_size, body3_size] (absolute values)
    """
    # Calculate indices for the 3 bars in the 222 pattern
    bar1_idx = setup_idx - 2  # First bar
    bar2_idx = setup_idx - 1  # Second bar
    bar3_idx = setup_idx      # Third bar (setup bar)
    
    # Calculate body sizes (absolute difference between open and close)
    body1_size = abs(close[bar1_idx] - open[bar1_idx])
    body2_size = abs(close[bar2_idx] - open[bar2_idx])
    body3_size = abs(close[bar3_idx] - open[bar3_idx])
    
    return np.array([body1_size, body2_size, body3_size], dtype=np.float64) 