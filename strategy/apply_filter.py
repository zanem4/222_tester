import numpy as np

def apply_filter(setups: np.ndarray, metrics_array: np.ndarray, constraints: dict) -> tuple:
    """
    Apply constraints to filter setups based on metric values
    
    Args:
        setups: Array of setup timestamps (positive=long, negative=short)
        metrics_array: Array of metrics for each setup (25 elements per setup - removed price_delta)
        constraints: Dictionary of constraint parameters (keys starting with 'const_')
    
    Returns:
        tuple: (filtered_setups, filtered_metrics_array)
    """
    if len(setups) == 0 or len(metrics_array) == 0:
        return setups, metrics_array
    
    # Create mask for valid setups (start with all True)
    valid_mask = np.ones(len(setups), dtype=bool)
    
    # Apply each constraint
    for constraint_key, constraint_value in constraints.items():
        if not constraint_key.startswith('const_'):
            continue
            
        # Extract metric name and constraint type from parameter name
        # Example: const_norm_upwick1_lower -> norm_upwick1, lower
        parts = constraint_key.split('_')
        if len(parts) < 3:
            continue
            
        metric_name = '_'.join(parts[1:-1])  # norm_upwick1
        constraint_type = parts[-1]  # lower or upper
        
        # Map metric names to their indices in the metrics array (updated for 25 elements)
        metric_index_map = {
            'sma': 1,
            'price_range': 2,
            'atr': 3,
            'body1_size': 4,
            'body2_size': 5,
            'body3_size': 6,
            'upwick1': 7,
            'downwick1': 8,
            'upwick2': 9,
            'downwick2': 10,
            'upwick3': 11,
            'downwick3': 12,
            'norm_sma': 13,
            'norm_price_range': 14,
            'norm_body1': 15,
            'norm_body2': 16,
            'norm_body3': 17,
            'norm_upwick1': 18,
            'norm_downwick1': 19,
            'norm_upwick2': 20,
            'norm_downwick2': 21,
            'norm_upwick3': 22,
            'norm_downwick3': 23
        }
        
        if metric_name not in metric_index_map:
            print(f"Warning: Unknown metric '{metric_name}' in constraint '{constraint_key}'")
            continue
            
        metric_index = metric_index_map[metric_name]
        metric_values = metrics_array[:, metric_index]
        
        # Apply constraint based on type
        if constraint_type == 'lower':
            valid_mask &= (metric_values >= constraint_value)
        elif constraint_type == 'upper':
            valid_mask &= (metric_values <= constraint_value)
        else:
            print(f"Warning: Unknown constraint type '{constraint_type}' in '{constraint_key}'")
            continue
    
    # Apply the mask to filter setups and metrics
    filtered_setups = setups[valid_mask]
    filtered_metrics = metrics_array[valid_mask]
    
    # Print filtering statistics
    original_count = len(setups)
    filtered_count = len(filtered_setups)
    removed_count = original_count - filtered_count
    
    if removed_count > 0:
        print(f"Filtering: {original_count} setups -> {filtered_count} setups (removed {removed_count})")
        
        # Print constraint details
        for constraint_key, constraint_value in constraints.items():
            if constraint_key.startswith('const_'):
                print(f"  Applied: {constraint_key} = {constraint_value}")
    
    return filtered_setups, filtered_metrics 