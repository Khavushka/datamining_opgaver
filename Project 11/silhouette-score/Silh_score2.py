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
