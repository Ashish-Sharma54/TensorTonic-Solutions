import numpy as np

def apply_homogeneous_transform(T: list, points: list) -> np.ndarray:
    """
    Returns transformed points with shape (3,) or (N, 3).
    """
    T = np.asarray(T, dtype=float)
    points = np.asarray(points, dtype=float)

    # Single point
    if points.ndim == 1:
        homogeneous = np.append(points, 1.0)
        transformed = T @ homogeneous
        return transformed[:3]

    # Batch of points
    ones = np.ones((points.shape[0], 1))
    homogeneous = np.hstack((points, ones))

    transformed = homogeneous @ T.T

    return transformed[:, :3]