def percent_change(series: list) -> list:
    """
    Returns the fractional change between consecutive values.
    """
    changes = []
    
    for i in range(1, len(series)):
        prev_val = series[i - 1]
        curr_val = series[i]
        
        if prev_val == 0:
            changes.append(0.0)
        else:
            fractional_change = (curr_val - prev_val) / prev_val
            changes.append(fractional_change)
            
    return changes
