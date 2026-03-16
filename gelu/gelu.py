import numpy as np
import math

def gelu(x):
    """
    Compute the Gaussian Error Linear Unit (exact version using erf).
    """
    x = np.asarray(x, dtype=float)
    erf_vec = np.vectorize(math.erf)
    return 0.5 * x * (1 + erf_vec(x / np.sqrt(2)))