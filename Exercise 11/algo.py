import pandas as pd
# import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.cluster import DBSCAN


# Load data into a Pandas dataframe
data = pd.read_csv("data.txt", header=None, delimiter="\t")

# Standardize the data
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)

# Apply K-Means clustering algorithm
kmeans = KMeans(n_clusters=3)
clusters = kmeans.fit_predict(data_scaled)

# DBSCAN
dbscan = DBSCAN(eps=0.3, min_samples=10)
y_pred = dbscan.fit_predict(data)

plt.scatter(data[:, 0], data[:, 1], c=y_pred, cmap='viridis')
plt.axis('equal')
plt.show()

# EM-clustering
em = GaussianMixture(n_components=3)
y_pred = em.fit_predict(data_scaled)
plt.scatter(data[:, 0], data[:, 1], c=y_pred, cmap='viridis')
plt.axis('equal')
plt.show()

# Visualize the clusters
plt.scatter(data.iloc[:,0], data.iloc[:,1], c=clusters, cmap="viridis")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("K-Means Clustering")
plt.show()
