import numpy as np

def pca_projection(X, k):
    """
    Project data onto the top-k principal components.
    """
    # Write code here
    X=np.array(X,dtype=float)
    X_centered=X-np.mean(X,axis=0)
    n=X_centered.shape[0]
    C=(X_centered.T@X_centered)/(n-1)
    eigenvalues, eigenvectors=np.linalg.eigh(C)
    idx=np.argsort(eigenvalues)[::-1]
    eigenvectors=eigenvectors[:,idx]
    W=eigenvectors[:,:k]
    X_proj=X_centered@W
    return X_proj