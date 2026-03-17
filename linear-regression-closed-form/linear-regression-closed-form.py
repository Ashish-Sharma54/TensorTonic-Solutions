import numpy as np

def linear_regression_closed_form(X, y):
    """
    Compute the optimal weight vector using the normal equation.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)

    Xt = X.T
    w = np.linalg.inv(Xt @ X) @ Xt @ y

    return w
    pass