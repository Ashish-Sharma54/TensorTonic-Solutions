import numpy as np

def entropy_node(y):
    """
    Compute entropy for a single node using stable logarithms.
    """
    y=np.asarray(y,dtype = float)
    # Write code here
    if y.size==0:
        return 0.0
    _, counts = np.unique(y, return_counts=True)
    probs= counts/counts.sum()
    probs=probs=probs[probs>0]
    
    return -np.sum(probs*np.log2(probs))
    pass