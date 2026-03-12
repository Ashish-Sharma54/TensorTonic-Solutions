def k_means_centroid_update(points, assignments, k):
    """
    Compute new centroids as the mean of assigned points.
    """
    # Write code here
    dim = len(points[0])  # dimensionality of points
    
    # initialize sums and counts
    sums = [[0.0] * dim for _ in range(k)]
    counts = [0] * k
    
    # accumulate sums
    for p, cluster in zip(points, assignments):
        for d in range(dim):
            sums[cluster][d] += p[d]
        counts[cluster] += 1
    
    # compute centroids
    centroids = []
    for j in range(k):
        if counts[j] == 0:
            centroids.append([0.0] * dim)
        else:
            centroids.append([sums[j][d] / counts[j] for d in range(dim)])
    
    return centroids