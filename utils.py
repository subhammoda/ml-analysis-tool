import pandas as pd

def detect_problem_type(y: pd.Series) -> str:
    """
    Automatically detect if the problem is classification or regression.
    
    Args:
        y (pd.Series): Target variable
        
    Returns:
        str: 'classification' or 'regression'
    """
    # Check if the target is categorical (object/string type)
    if y.dtype == 'object' or y.dtype.name == 'category':
        return 'classification'
    
    # Check if the target is integer with reasonable number of unique values
    if y.dtype in ['int64', 'int32', 'int16', 'int8']:
        unique_count = len(y.unique())
        # If it's integer and has reasonable number of unique values, likely classification
        if unique_count <= 50:  # Increased threshold for multi-class
            return 'classification'
    
    # Check if the target appears to be categorical despite being numeric
    # (e.g., encoded categorical variables)
    if y.dtype in ['float64', 'float32', 'float16']:
        unique_count = len(y.unique())
        # If it has few unique values relative to total size, likely classification
        if unique_count <= 50 and unique_count < len(y) * 0.1:  # Less than 10% unique values
            return 'classification'
    
    # Default to regression for continuous numeric variables
    return 'regression'

 