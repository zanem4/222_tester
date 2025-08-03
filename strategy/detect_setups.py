# 222 pattern detection

import numpy as np

def detect_setups(time, high, low, close, open_array):
    # Slice arrays for vector comparisons
    h1, h2, h3 = high[:-2], high[1:-1], high[2:]
    c1, c2, c3 = close[:-2], close[1:-1], close[2:]
    l1, l2, l3 = low[:-2], low[1:-1], low[2:]
    o1, o2, o3 = open_array[:-2], open_array[1:-1], open_array[2:]

    # Long setup: strictly increasing highs, closes, and lows + all green candles
    long_mask = ((h3 > h2) & (h2 > h1) & (c3 > c2) & (c2 > c1) & (l3 > l2) & (l2 > l1) & 
                 (c1 > o1) & (c2 > o2) & (c3 > o3))  # All 3 candles must be green

    # Short setup: strictly decreasing highs, closes, and lows + all red candles
    short_mask = ((h3 < h2) & (h2 < h1) & (c3 < c2) & (c2 < c1) & (l3 < l2) & (l2 < l1) & 
                  (c1 < o1) & (c2 < o2) & (c3 < o3))  # All 3 candles must be red

    # Get indices for both types
    long_indices = np.where(long_mask)[0] + 2
    short_indices = np.where(short_mask)[0] + 2
    
    # REMOVED FILTERING - For parameter optimization, we want all setups
    # Apply filtering - minimum 5 bars between setups
    # if len(long_indices) > 0:
    #     filtered_long = [long_indices[0]]
    #     for idx in long_indices[1:]:
    #         if idx - filtered_long[-1] >= 5:  # At least 5 bars apart
    #             filtered_long.append(idx)
    #     long_indices = np.array(filtered_long)
    
    # if len(short_indices) > 0:
    #     filtered_short = [short_indices[0]]
    #     for idx in short_indices[1:]:
    #         if idx - filtered_short[-1] >= 5:  # At least 5 bars apart
    #             filtered_short.append(idx)
    #     short_indices = np.array(filtered_short)
    
    # Combine with sign convention: positive = long, negative = short
    long_times = time[long_indices]
    short_times = -time[short_indices]  # Negative for short setups
    
    # Use hstack instead of concatenate
    all_setups = np.hstack((long_times, short_times))
    
    # Sort by absolute value to maintain chronological order
    return np.sort(all_setups)