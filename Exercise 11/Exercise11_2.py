# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# from sklearn.cluster import KMeans, DBSCAN
# from sklearn.mixture import GaussianMixture
# from sklearn.preprocessing import StandardScaler
# from sklearn.metrics import silhouette_score

# # Load Wine dataset
# url = "http://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data"
# wine_df = pd.read_csv(url, header=None)
# X = wine_df.iloc[:, 1:].values

# # Preprocess data
# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X)

# # Set parameters for clustering algorithms
# kmeans = KMeans(n_clusters=3, random_state=42)
# dbscan = DBSCAN(eps=1, min_samples=5)
# gmm = GaussianMixture(n_components=3, random_state=42)

# # Fit clustering algorithms
# kmeans.fit(X_scaled)
# dbscan.fit(X_scaled)
# gmm.fit(X_scaled)

# # Get cluster labels
# kmeans_labels = kmeans.labels_
# dbscan_labels = dbscan.labels_
# gmm_labels = gmm.predict(X_scaled)

# # Calculate Silhouette scores
# kmeans_score = silhouette_score(X_scaled, kmeans_labels)
# dbscan_score = silhouette_score(X_scaled, dbscan_labels) if -1 not in dbscan_labels else np.nan
# gmm_score = silhouette_score(X_scaled, gmm_labels)

# # Plot results
# fig, ax = plt.subplots(1, 3, figsize=(12, 4))
# ax[0].scatter(X_scaled[:, 0], X_scaled[:, 1], c=kmeans_labels)
# ax[0].set_title(f"KMeans\nSilhouette score: {kmeans_score:.2f}")
# ax[1].scatter(X_scaled[:, 0], X_scaled[:, 1], c=dbscan_labels)
# ax[1].set_title(f"DBSCAN\nSilhouette score: {dbscan_score:.2f}")
# ax[2].scatter(X_scaled[:, 0], X_scaled[:, 1], c=gmm_labels)
# ax[2].set_title(f"EM-Clustering\nSilhouette score: {gmm_score:.2f}")
# plt.show()

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

# Load Wine dataset
url = "http://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data"
wine_df = pd.read_csv(url, header=None)
X = wine_df.iloc[:, 1:].values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Parameters for clustering algorithms
kmeans = KMeans(n_clusters=3, random_state=42)
dbscan = DBSCAN(eps=2, min_samples=3)
gmm = GaussianMixture(n_components=3, random_state=42)

# PCA- for at reducere dimensionalitet
pca = PCA(n_components=3)
X_pca = pca.fit_transform(X_scaled)

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
fig= plt.figure(figsize=(15, 5))
# K-means clustering
ax = fig.add_subplot(131, projection='3d')
ax.scatter(X_pca[:, 0], X_pca[:, 1], X_pca[:, 2], c=kmeans_labels)
ax.set_title(f"K-means (score={kmeans_score:.3f})")

# DBSCAN clustering
ax = fig.add_subplot(132, projection='3d')
ax.scatter(X_pca[:, 0], X_pca[:, 1], X_pca[:, 2], c=dbscan_labels)
ax.set_title(f"DBSCAN (score={dbscan_score:.3f})")

# EM clustering
ax = fig.add_subplot(133, projection='3d')
ax.scatter(X_pca[:, 0], X_pca[:, 1], X_pca[:, 2], c=gmm_labels)
ax.set_title(f"EM-clustering (score={gmm_score:.3f})")
plt.show()

# Her printer vi silhouette-score ud
print("Silhouette-score for K-means: ", kmeans_score)
print("Silhouette-score for DBSCAN: ", dbscan_score)
print("Silhouette-score for EM-clustering: ", gmm_score)