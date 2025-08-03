# param loading and verification

import json
from datetime import datetime

def validate_date_format(date_str, param_name):
    """
    Validate that a date string is in the correct ISO format
    
    Args:
        date_str: Date string to validate
        param_name: Name of the parameter for error messages
    
    Returns:
        bool: True if valid, False otherwise
    """
    if not date_str:
        return True  # Empty dates are allowed (optional parameters)
    
    try:
        datetime.fromisoformat(date_str)
        return True
    except ValueError:
        print(f"Error: Invalid {param_name} format '{date_str}'. Expected format: YYYY-MM-DDTHH:MM:SS")
        return False

def load_parameters(path: str = "params.json") -> dict:
    
    with open(path, "r") as f:
        params = json.load(f)

    # Validate date parameters if present
    if "start_date" in params:
        if not validate_date_format(params["start_date"], "start_date"):
            print("Removing invalid start_date parameter")
            params.pop("start_date")
    
    if "stop_date" in params:
        if not validate_date_format(params["stop_date"], "stop_date"):
            print("Removing invalid stop_date parameter")
            params.pop("stop_date")
    
    # Validate that start_date is before stop_date if both are present
    if "start_date" in params and "stop_date" in params:
        try:
            start_dt = datetime.fromisoformat(params["start_date"])
            stop_dt = datetime.fromisoformat(params["stop_date"])
            if start_dt >= stop_dt:
                print("Error: start_date must be before stop_date")
                print(f"start_date: {params['start_date']}")
                print(f"stop_date: {params['stop_date']}")
                # Remove both dates to prevent issues
                params.pop("start_date")
                params.pop("stop_date")
        except ValueError:
            # This shouldn't happen if validate_date_format passed, but just in case
            pass

    # Handle constraint parameters
    if params.get("use_constraints", False):
        # Extract constraint parameters (those starting with 'const_')
        constraint_params = {}
        constraint_keys_to_remove = []
        
        for key, value in params.items():
            if key.startswith('const_'):
                constraint_params[key] = value
                constraint_keys_to_remove.append(key)
        
        # Store constraint parameters separately
        params['constraint_params'] = constraint_params
        
        # Remove constraint keys from main params to avoid confusion
        for key in constraint_keys_to_remove:
            params.pop(key)
            
        print(f"Loaded {len(constraint_params)} constraint parameters")
        for key, value in constraint_params.items():
            print(f"  {key}: {value}")
    else:
        # If constraints are disabled, remove constraint-related keys
        constraint_keys = [
            "min_body_atr",
            "max_body_atr",
            "min_delta_atr",
            "max_delta_atr",
            # Add other constraint keys here
        ]
        for key in constraint_keys:
            params.pop(key, None)  # Safe removal if key doesn't exist
        
        # Also remove any constraint parameters
        constraint_keys_to_remove = [key for key in params.keys() if key.startswith('const_')]
        for key in constraint_keys_to_remove:
            params.pop(key)

    return params
# p = load_parameters(r"C:\Users\zanem\OneDrive\Desktop\Personal\222_Backtester\backtester\utils\parameters.json")