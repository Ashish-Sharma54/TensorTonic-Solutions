import numpy as np

def rnn_step_forward(x_t: list, h_prev: list, Wx: list, Wh: list, b: list) -> np.ndarray:
    """
    Computes a single forward step of a tanh RNN cell.
    Returns a NumPy array with shape (H,).
    """
    # Convert input lists to NumPy arrays
    x_arr = np.array(x_t)
    h_arr = np.array(h_prev)
    Wx_arr = np.array(Wx)
    Wh_arr = np.array(Wh)
    b_arr = np.array(b)
    
    # Compute affine transformation: x_t * Wx + h_{t-1} * Wh + b
    # np.dot or @ performs matrix-vector multiplication correctly
    a_t = np.dot(x_arr, Wx_arr) + np.dot(h_arr, Wh_arr) + b_arr
    
    # Apply the element-wise tanh activation function
    h_t = np.tanh(a_t)
    
    return h_t
