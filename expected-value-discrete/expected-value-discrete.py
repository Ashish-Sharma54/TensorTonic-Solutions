import numpy as np

def expected_value_discrete(x, p):
    """
    Returns: float expected value
    """
    x = np.asarray(x, dtype=float)
    p = np.asarray(p, dtype=float)

    # Check shapes
    if x.shape != p.shape:
        raise ValueError("x and p must have the same shape")

    # Check probabilities sum to 1
    if not np.isclose(np.sum(p), 1.0, atol=1e-6):
        raise ValueError("Probabilities must sum to 1")

    # Expected value: sum(x_i * p_i)
    return float(np.sum(x * p))