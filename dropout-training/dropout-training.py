import numpy as np

def dropout(
    x: list,
    p: float = 0.5,
    rng: np.random.Generator = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Apply dropout to input x with probability p.
    Return (output, dropout_pattern).
    """
    x = np.asarray(x, dtype=float)

    if rng is None:
        random_values = np.random.random(x.shape)
    else:
        random_values = rng.random(x.shape)

    scale = 1.0 / (1.0 - p)

    # Keep if random value >= p
    mask = (random_values >= p).astype(float)

    # Scaled mask
    dropout_pattern = mask * scale

    # Apply dropout
    output = x * dropout_pattern

    return output, dropout_pattern