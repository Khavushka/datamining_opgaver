import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, DBSCAN
from sklearn.mixture import GaussianMixture

# Load data from URL
url = "https://cs.joensuu.fi/sipu/datasets/spiral.txt"
data = np.loadtxt(url)

# K-means clustering
kmeans = KMeans(n_clusters=3, random_state=0).fit(data)
kmeans_labels = kmeans.labels_

# DBSCAN clustering
dbscan = DBSCAN(eps=1, min_samples=3).fit(data)
dbscan_labels = dbscan.labels_

# EM-clustering
em = GaussianMixture(n_components=3, random_state=0).fit(data)
em_labels = em.predict(data)

# Plot results
plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
plt.scatter(data[:, 0], data[:, 1], c=kmeans_labels)
plt.title("K-means clustering")

plt.subplot(1, 3, 2)
plt.scatter(data[:, 0], data[:, 1], c=dbscan_labels)
plt.title("DBSCAN clustering")

plt.subplot(1, 3, 3)
plt.scatter(data[:, 0], data[:, 1], c=em_labels)
plt.title("EM-clustering")

plt.show()