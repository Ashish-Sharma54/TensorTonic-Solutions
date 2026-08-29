import numpy as np

def gru_cell_forward(x: list, h_prev: list, params: dict) -> np.ndarray:
    """
    Returns the updated hidden state as a NumPy array matching the shape of h_prev.
    """

    x = np.asarray(x)
    h_prev = np.asarray(h_prev)

    # Parameters
    Wz = params["Wz"]
    Wr = params["Wr"]
    Wh = params["Wh"]

    Uz = params["Uz"]
    Ur = params["Ur"]
    Uh = params["Uh"]

    bz = params["bz"]
    br = params["br"]
    bh = params["bh"]

    # Sigmoid function
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    # 1. Update gate
    z = sigmoid(x @ Wz + h_prev @ Uz + bz)

    # 2. Reset gate
    r = sigmoid(x @ Wr + h_prev @ Ur + br)

    # 3. Candidate hidden state
    h_tilde = np.tanh(
        x @ Wh + (r * h_prev) @ Uh + bh
    )

    # 4. New hidden state
    h = (1 - z) * h_prev + z * h_tilde

    return h