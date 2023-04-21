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
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Parameters for clustering algorithms
kmeans = KMeans(n_clusters=3, random_state=42)
dbscan = DBSCAN(eps=1, min_samples=5)
gmm = GaussianMixture(n_components=3, random_state=42)

kmeans.fit(X_scaled)
dbscan.fit(X_scaled)
gmm.fit(X_scaled)

# Get cluster labels
kmeans_labels = kmeans.labels_
dbscan_labels = dbscan.labels_
gmm_labels = gmm.predict(X_scaled)

# Beregner Silhouette scores
kmeans_score = silhouette_score(X_scaled, kmeans_labels)
dbscan_score = silhouette_score(X_scaled, dbscan_labels) if -1 not in dbscan_labels else np.nan
gmm_score = silhouette_score(X_scaled, gmm_labels)

# Plot results
fig, ax = plt.subplots(1, 3, figsize=(12, 4))
ax[0].scatter(X_scaled[:, 0], X_scaled[:, 1], c=kmeans_labels)
ax[0].set_title(f"KMeans\nSilhouette score: {kmeans_score:.2f}")
ax[1].scatter(X_scaled[:, 0], X_scaled[:, 1], c=dbscan_labels)
ax[1].set_title(f"DBSCAN\nSilhouette score: {dbscan_score:.2f}")
ax[2].scatter(X_scaled[:, 0], X_scaled[:, 1], c=gmm_labels)
ax[2].set_title(f"EM-Clustering\nSilhouette score: {gmm_score:.2f}")
plt.show()