import numpy as np

def cross_entropy_loss(y_true: list[int], y_pred: list[list[float]]) -> float:
    """
    Return the mean multiclass cross-entropy loss.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred, dtype=float)

    # Select probability of the correct class for each sample
    correct_probs = y_pred[np.arange(len(y_true)), y_true]

    # Cross-entropy loss
    loss = -np.mean(np.log(correct_probs))

    return float(loss)