import numpy as np
from sklearn.cluster import KMeans

# Generate some random data points
np.random.seed(0)
X = np.random.rand(100, 2)

# Specify the number of clusters (K)
K = 3

# Create a K-means object and fit the data
kmeans = KMeans(n_clusters=K, random_state=0)
kmeans.fit(X)

# Get the cluster labels and cluster centers
cluster_labels = kmeans.labels_
cluster_centers = kmeans.cluster_centers_

# Print the cluster labels and cluster centers
print("Cluster Labels:")
print(cluster_labels)
print("\nCluster Centers:")
print(cluster_centers)
