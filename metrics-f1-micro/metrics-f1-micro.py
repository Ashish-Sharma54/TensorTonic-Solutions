def f1_micro(y_true, y_pred) -> float:
    """
    Compute micro-averaged F1 for multi-class integer labels.
    """
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")

    if len(y_true) == 0:
        return 0.0

    tp = sum(true == pred for true, pred in zip(y_true, y_pred))

    fp = len(y_true) - tp
    fn = len(y_true) - tp

    denominator = 2 * tp + fp + fn

    if denominator == 0:
        return 0.0

    return float(2 * tp / denominator)