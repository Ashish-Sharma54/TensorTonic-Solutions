import numpy as np

def positional_encoding(seq_len, d_model, base=10000.0):
    """
    Return PE of shape (seq_len, d_model) using sin/cos formulation.
    Odd d_model -> last column is sin.
    """
    positions = np.arange(seq_len)[:, None]

    # Pair indices: i = 0, 1, 2, ...
    i = np.arange((d_model + 1) // 2)

    # divisor = base^(2i / d_model)
    div_term = base ** (2 * i / d_model)

    pe = np.zeros((seq_len, d_model), dtype=float)

    # Even columns: sin
    pe[:, 0::2] = np.sin(positions / div_term)

    # Odd columns: cos
    pe[:, 1::2] = np.cos(
        positions / div_term[:d_model // 2]
    )

    return pe