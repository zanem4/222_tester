import numpy as np
from strategy.atr import calculate_atr

def calculate_metrics(setup_time: np.float64, time_array: np.ndarray, high: np.ndarray, 
                     low: np.ndarray, close: np.ndarray, open: np.ndarray, 
                     sma_period: int = 30, atr_period: int = 14) -> np.ndarray:
    """
    Calculate trading metrics for a given setup timestamp
    
    Args:
        setup_time: UNIX timestamp of setup (positive=long, negative=short)
        time_array: Array of UNIX timestamps
        high: High prices array
        low: Low prices array
        close: Close prices array
        open: Open prices array
        sma_period: Period for SMA calculation (default 30)
        atr_period: Period for ATR calculation (default 14)
    
    Returns:
        numpy array: [setup_idx, sma, price_range, atr, body1_size, body2_size, body3_size, upwick1, downwick1, upwick2, downwick2, upwick3, downwick3, norm_sma, norm_price_range, norm_body1, norm_body2, norm_body3, norm_upwick1, norm_downwick1, norm_upwick2, norm_downwick2, norm_upwick3, norm_downwick3]
    """
    # Find setup index
    setup_idx = np.where(time_array == abs(setup_time))[0]
    if len(setup_idx) == 0:
        return np.array([np.nan] * 25)  # Reduced from 26 to 25
    setup_idx = setup_idx[0]
    
    # Check if we have enough data for the 222 pattern
    if setup_idx < 2:
        return np.array([np.nan] * 25)  # Reduced from 26 to 25
    
    # Get the 3 bars of the 222 pattern
    bar1_idx = setup_idx - 2
    bar2_idx = setup_idx - 1
    bar3_idx = setup_idx
    
    # Calculate basic metrics (removed price_delta)
    price_range = np.max(high[bar1_idx:bar3_idx+1]) - np.min(low[bar1_idx:bar3_idx+1])
    
    # Calculate SMA
    sma_start = max(0, setup_idx - sma_period + 1)
    sma_window = close[sma_start:setup_idx + 1]
    sma = np.mean(sma_window)
    
    # Calculate ATR
    atr_start = max(0, setup_idx - atr_period + 1)
    atr_window_high = high[atr_start:setup_idx + 1]
    atr_window_low = low[atr_start:setup_idx + 1]
    atr_window_close = close[atr_start:setup_idx + 1]
    atr = calculate_atr(atr_window_high, atr_window_low, atr_window_close, atr_period)
    
    # If ATR is 0 or very small, use a fallback calculation
    if atr <= 0.00001:
        # Use simple price range as fallback ATR
        atr = np.mean(high[atr_start:setup_idx + 1] - low[atr_start:setup_idx + 1])
        if atr <= 0.00001:
            atr = 0.0001  # Minimum ATR value
    
    # Calculate body sizes
    body1_size = abs(close[bar1_idx] - open[bar1_idx])
    body2_size = abs(close[bar2_idx] - open[bar2_idx])
    body3_size = abs(close[bar3_idx] - open[bar3_idx])
    
    # Calculate wick sizes
    upwick1 = high[bar1_idx] - np.maximum(close[bar1_idx], open[bar1_idx])
    downwick1 = np.minimum(close[bar1_idx], open[bar1_idx]) - low[bar1_idx]
    upwick2 = high[bar2_idx] - np.maximum(close[bar2_idx], open[bar2_idx])
    downwick2 = np.minimum(close[bar2_idx], open[bar2_idx]) - low[bar2_idx]
    upwick3 = high[bar3_idx] - np.maximum(close[bar3_idx], open[bar3_idx])
    downwick3 = np.minimum(close[bar3_idx], open[bar3_idx]) - low[bar3_idx]
    
    # Calculate normalized metrics
    # Use a minimum ATR threshold to avoid division by very small numbers
    min_atr_threshold = np.float64(0.00001)  # Minimum ATR value for normalization
    effective_atr = np.maximum(atr, min_atr_threshold)
    
    norm_sma = sma  # Keep SMA as absolute price level, don't normalize
    norm_price_range = price_range / effective_atr
    norm_body1 = body1_size / effective_atr
    norm_body2 = body2_size / effective_atr
    norm_body3 = body3_size / effective_atr
    norm_upwick1 = upwick1 / effective_atr
    norm_downwick1 = downwick1 / effective_atr
    norm_upwick2 = upwick2 / effective_atr
    norm_downwick2 = downwick2 / effective_atr
    norm_upwick3 = upwick3 / effective_atr
    norm_downwick3 = downwick3 / effective_atr
    
    # Return exactly 25 elements (removed price_delta and norm_price_delta)
    return np.array([
        float(setup_idx), sma, price_range, atr,
        body1_size, body2_size, body3_size,
        upwick1, downwick1, upwick2, downwick2, upwick3, downwick3,
        norm_sma, norm_price_range, norm_body1, norm_body2, norm_body3,
        norm_upwick1, norm_downwick1, norm_upwick2, norm_downwick2, norm_upwick3, norm_downwick3
    ], dtype=np.float64)