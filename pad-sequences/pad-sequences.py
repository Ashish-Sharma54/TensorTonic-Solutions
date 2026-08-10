import numpy as np

def pad_sequences(seqs, pad_value=0, max_len=None):
    """
    Returns: np.ndarray of shape (N, L)
    """
    # Empty input
    if len(seqs) == 0:
        return np.empty((0, 0), dtype=int)

    # Automatically find maximum length
    if max_len is None:
        max_len = max(len(seq) for seq in seqs)

    # Create output filled with pad_value
    result = np.full((len(seqs), max_len), pad_value, dtype=int)

    # Copy values, truncating if necessary
    for i, seq in enumerate(seqs):
        length = min(len(seq), max_len)
        result[i, :length] = seq[:length]

    return result
    pass