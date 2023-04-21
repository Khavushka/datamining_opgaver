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
dbscan = DBSCAN(eps=2, min_samples=3)
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

'''Silhouette-score til second-dataset
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# Load Wine dataset
url = "http://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data"
wine_df = pd.read_csv(url, header=None)
X = wine_df.iloc[:, 1:].values

# Preprocess data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Set parameters for clustering algorithms
kmeans = KMeans(n_clusters=3, random_state=42)   # number of clusters = 3, random seed = 42
dbscan = DBSCAN(eps=1, min_samples=5)            # neighborhood radius = 1, minimum number of samples in a neighborhood = 5
gmm = GaussianMixture(n_components=3, random_state=42)   # number of Gaussian components = 3, random seed = 42

# Fit clustering algorithms
kmeans.fit(X_scaled)
dbscan.fit(X_scaled)
gmm.fit(X_scaled)

# Get cluster labels
kmeans_labels = kmeans.labels_
dbscan_labels = dbscan.labels_
gmm_labels = gmm.predict(X_scaled)

# Compute quality measures
kmeans_silhouette = silhouette_score(X_scaled, kmeans_labels)

dbscan_silhouette = silhouette_score(X_scaled, dbscan_labels)

gmm_silhouette = silhouette_score(X_scaled, gmm_labels)

print("K-Means Clustering")
print("Silhouette Coefficient:", kmeans_silhouette)
print()

print("DBSCAN Clustering")
print("Silhouette Coefficient:", dbscan_silhouette)
print()

print("EM Clustering")
print("Silhouette Coefficient:", gmm_silhouette)

'''