import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score

# Load the spiral dataset
url = "https://cs.joensuu.fi/sipu/datasets/spiral.txt"
data = np.loadtxt(url)

# K-means clustering
kmeans = KMeans(n_clusters=3, random_state=0)
kmeans_labels = kmeans.fit_predict(data)
kmeans_score = silhouette_score(data, kmeans_labels)

# DBSCAN clustering
dbscan = DBSCAN(eps=1, min_samples=3)
dbscan_labels = dbscan.fit_predict(data)
dbscan_score = silhouette_score(data, dbscan_labels)

# EM-clustering
em = GaussianMixture(n_components=3, random_state=0)
em_labels = em.fit_predict(data)
em_score = silhouette_score(data, em_labels)

# Print silhouette coefficients for hver clustering solution
print(f"K-means score: {kmeans_score:.3f}")
print(f"DBSCAN score: {dbscan_score:.3f}")
print(f"EM-clustering score: {em_score:.3f}")

# Plot the clustering results
fig, ax = plt.subplots(1, 3, figsize=(12, 4))

ax[0].scatter(data[:, 0], data[:, 1], c=kmeans_labels)
ax[0].set_title(f"K-means (score={kmeans_score:.3f})")

ax[1].scatter(data[:, 0], data[:, 1], c=dbscan_labels)
ax[1].set_title(f"DBSCAN (score={dbscan_score:.3f})")

ax[2].scatter(data[:, 0], data[:, 1], c=em_labels)
ax[2].set_title(f"EM-clustering (score={em_score:.3f})")

plt.show()