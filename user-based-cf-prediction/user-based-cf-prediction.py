def user_based_cf_prediction(similarities: list, ratings: list) -> float:
    """
    Returns the positive-similarity weighted rating prediction.
    """
    weighted_sum = 0.0
    similarity_sum = 0.0
    
    # Iterate through pairs of similarities and ratings
    for sim, rat in zip(similarities, ratings):
        # Only consider users with positive similarity
        if sim > 0:
            weighted_sum += sim * rat
            similarity_sum += sim
            
    # Return 0.0 if no user has positive similarity to avoid division by zero
    if similarity_sum == 0:
        return 0.0
        
    # Calculate the prediction and round to 6 decimal places
    return round(weighted_sum / similarity_sum, 6)
    pass