import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# Load Wine dataset
url = "http://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data"
wine_df = pd.read_csv(url, header=None)
X = wine_df.iloc[:, 1:].values

# Preprocess data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Perform PCA to reduce dimensionality
pca = PCA(n_components=3)
X_pca = pca.fit_transform(X_scaled)

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

# Plot 3D scatter plots
fig = plt.figure(figsize=(15, 5))

# K-means clustering
ax = fig.add_subplot(131, projection='3d')
ax.scatter(X_pca[:, 0], X_pca[:, 1], X_pca[:, 2], c=kmeans_labels)
ax.set_title("K-Means Clustering")

# DBSCAN clustering
ax = fig.add_subplot(132, projection='3d')
ax.scatter(X_pca[:, 0], X_pca[:, 1], X_pca[:, 2], c=dbscan_labels)
ax.set_title("DBSCAN Clustering")

# EM clustering
ax = fig.add_subplot(133, projection='3d')
ax.scatter(X_pca[:, 0], X_pca[:, 1], X_pca[:, 2], c=gmm_labels)
ax.set_title("EM Clustering")

plt.show()
