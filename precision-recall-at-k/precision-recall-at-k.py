def precision_recall_at_k(recommended, relevant, k):
    """
    Compute precision@k and recall@k for a recommendation list.
    """
    # Write code here
    top_k = recommended[:k]
    relevant_set = set(relevant)
    hits = sum(item in relevant_set for item in top_k)
    precision = hits /k
    recall = hits / len(relevant)
    return [float(precision),float(recall)]