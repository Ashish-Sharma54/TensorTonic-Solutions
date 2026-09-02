import numpy as np

def contrastive_loss(a: list, b: list, y: list, margin: float = 1.0, reduction: str = "mean") -> float:
    """ Returns the loss as a float. """
    # Convert inputs to NumPy arrays
    a = np.array(a, dtype=float)
    b = np.array(b, dtype=float)
    y = np.array(y, dtype=float)
    
    # Handle 1D inputs by reshaping them to a single row (2D)
    if a.ndim == 1:
        a = a.reshape(1, -1)
    if b.ndim == 1:
        b = b.reshape(1, -1)
    if y.ndim == 0:
        y = y.reshape(1)
        
    # Compute Euclidean distance per pair along the feature axis
    d = np.linalg.norm(a - b, axis=1)
    
    # Compute pair loss: l_i = y_i * d_i^2 + (1 - y_i) * max(0, margin - d_i)^2
    loss = y * (d ** 2) + (1 - y) * np.maximum(0, margin - d) ** 2
    
    # Apply reduction
    if reduction == "mean":
        return float(np.mean(loss))
    elif reduction == "sum":
        return float(np.sum(loss))
    else:
        raise ValueError("reduction must be either 'mean' or 'sum'")
