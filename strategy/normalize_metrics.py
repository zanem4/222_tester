import numpy as np

def normalize_metrics(price_delta: np.float64, sma: np.float64, price_range: np.float64, 
                     body1_size: np.float64, body2_size: np.float64, body3_size: np.float64,
                     upwick1: np.float64, downwick1: np.float64, upwick2: np.float64, downwick2: np.float64,
                     upwick3: np.float64, downwick3: np.float64, atr: np.float64) -> np.ndarray:
    """
    Normalize price-based metrics by ATR for inter-asset comparison
    
    Args:
        price_delta: Price delta from 3-bar extrema
        sma: Simple moving average (absolute price level - should NOT be normalized)
        price_range: Max high - min low over window
        body1_size: Body size of first bar in 222 pattern
        body2_size: Body size of second bar in 222 pattern
        body3_size: Body size of third bar in 222 pattern
        upwick1: Upper wick of first bar
        downwick1: Lower wick of first bar
        upwick2: Upper wick of second bar
        downwick2: Lower wick of second bar
        upwick3: Upper wick of third bar
        downwick3: Lower wick of third bar
        atr: Average True Range
    
    Returns:
        numpy array: [norm_price_delta, norm_sma, norm_price_range, norm_body1, norm_body2, norm_body3, norm_upwick1, norm_downwick1, norm_upwick2, norm_downwick2, norm_upwick3, norm_downwick3]
    """
    # Avoid division by zero
    if atr <= 0:
        return np.array([np.nan] * 12, dtype=np.float64)
    
    # Normalize each metric by ATR (except SMA which is an absolute price level)
    norm_price_delta = price_delta / atr
    norm_sma = sma  # Keep SMA as absolute price level, don't normalize
    norm_price_range = price_range / atr
    norm_body1 = body1_size / atr
    norm_body2 = body2_size / atr
    norm_body3 = body3_size / atr
    norm_upwick1 = upwick1 / atr
    norm_downwick1 = downwick1 / atr
    norm_upwick2 = upwick2 / atr
    norm_downwick2 = downwick2 / atr
    norm_upwick3 = upwick3 / atr
    norm_downwick3 = downwick3 / atr
    
    return np.array([norm_price_delta, norm_sma, norm_price_range, norm_body1, norm_body2, norm_body3,
                    norm_upwick1, norm_downwick1, norm_upwick2, norm_downwick2, norm_upwick3, norm_downwick3], dtype=np.float64) 