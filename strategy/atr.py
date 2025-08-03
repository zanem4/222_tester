import numpy as np
import numba

@numba.njit
def calculate_atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int) -> np.float64:
    
    # High - Low
    hl = high - low
    
    # High - Previous Close (skip first bar)
    hc = np.abs(high[1:] - close[:-1])
    
    # Low - Previous Close (skip first bar)
    lc = np.abs(low[1:] - close[:-1])
    
    # True Range = max of the three (starting from second bar)
    true_range = np.maximum(hl[1:], np.maximum(hc, lc))
    
    # Calculate Simple Moving Average of True Range
    if len(true_range) < period:
        return np.float64(0.0)
    
    # Simple Moving Average
    atr_values = []
    for i in range(period - 1, len(true_range)):
        atr_values.append(np.mean(true_range[i - period + 1:i + 1]))
    
    # Return final ATR value with 5 decimal places
    final_atr = np.float64(atr_values[-1]) if atr_values else np.float64(0.0)
    return np.round(final_atr, 5)