import numpy as np

def train_kmeans(X, k, max_iterations):
    num_datapoints, num_features = X.shape
    
    # randomly initialize cluster centroids
    centroids = np.random.rand(k, num_features)
    
    for i in range(max_iterations):
        # calculate distances between datapoints and centroids
        distances = np.array([np.linalg.norm(X - centroid, axis=1) for centroid in centroids])
        
        # assign each datapoint to the closest centroid
        clusters = np.argmin(distances, axis=0)
        
        # recalculate the centroids for each cluster
        for j in range(k):
            centroids[j] = np.mean(X[clusters == j], axis=0)
    
    return centroids, clusters
