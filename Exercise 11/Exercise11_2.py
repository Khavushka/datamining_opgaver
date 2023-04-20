# 1-solution ------------------
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.cluster import KMeans, DBSCAN
# from sklearn.mixture import GaussianMixture

# # Load the wine dataset
# url = "https://cs.joensuu.fi/sipu/datasets/wine.txt"
# data = pd.read_csv(url, delimiter='\t', header=None, names=['x', 'y', 'z', 'label'])
# X = data.drop('label', axis=1)

# # Visualize the dataset
# fig = plt.figure(figsize=(8, 6))
# ax = fig.add_subplot(projection='3d')
# ax.scatter(X['x'], X['y'], X['z'], c=data['label'], cmap='viridis')
# ax.set_xlabel('x')
# ax.set_ylabel('y')
# ax.set_zlabel('z')
# plt.show()

# # Apply K-means clustering
# kmeans = KMeans(n_clusters=3, random_state=42)
# kmeans_labels = kmeans.fit_predict(X)

# # Apply DBSCAN clustering
# dbscan = DBSCAN(eps=0.5, min_samples=5)
# dbscan_labels = dbscan.fit_predict(X)

# # Apply EM-clustering
# em = GaussianMixture(n_components=3, random_state=42)
# em_labels = em.fit_predict(X)

# # Visualize the clustering results
# fig = plt.figure(figsize=(16, 6))

# ax1 = fig.add_subplot(131, projection='3d')
# ax1.scatter(X['x'], X['y'], X['z'], c=kmeans_labels, cmap='viridis')
# ax1.set_xlabel('x')
# ax1.set_ylabel('y')
# ax1.set_zlabel('z')
# ax1.set_title('K-means')

# ax2 = fig.add_subplot(132, projection='3d')
# ax2.scatter(X['x'], X['y'], X['z'], c=dbscan_labels, cmap='viridis')
# ax2.set_xlabel('x')
# ax2.set_ylabel('y')
# ax2.set_zlabel('z')
# ax2.set_title('DBSCAN')

# ax3 = fig.add_subplot(133, projection='3d')
# ax3.scatter(X['x'], X['y'], X['z'], c=em_labels, cmap='viridis')
# ax3.set_xlabel('x')
# ax3.set_ylabel('y')
# ax3.set_zlabel('z')
# ax3.set_title('EM-clustering')

# plt.show()


import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, DBSCAN
from sklearn.mixture import GaussianMixture
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

# Load data
url_data = pd.read_csv('urls.csv')
url_data = url_data.dropna()
url_data = url_data[['url_length', 'num_special_chars', 'num_numbers']]

# Scale data
url_data = (url_data - url_data.mean()) / url_data.std()

# K-means clustering
kmeans = KMeans(n_clusters=3, random_state=0).fit(url_data)
url_data['kmeans_cluster'] = kmeans.labels_

# DBSCAN clustering
dbscan = DBSCAN(eps=0.3, min_samples=5).fit(url_data)
url_data['dbscan_cluster'] = dbscan.labels_

# EM-clustering
gmm = GaussianMixture(n_components=3, random_state=0).fit(url_data)
url_data['gmm_cluster'] = gmm.predict(url_data)

# Plot clusters
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(url_data['url_length'], url_data['num_special_chars'], url_data['num_numbers'], c=url_data['kmeans_cluster'])
ax.set_title('K-means Clustering')
ax.set_xlabel('URL Length')
ax.set_ylabel('Number of Special Characters')
ax.set_zlabel('Number of Numbers')
plt.show()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(url_data['url_length'], url_data['num_special_chars'], url_data['num_numbers'], c=url_data['dbscan_cluster'])
ax.set_title('DBSCAN Clustering')
ax.set_xlabel('URL Length')
ax.set_ylabel('Number of Special Characters')
ax.set_zlabel('Number of Numbers')
plt.show()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(url_data['url_length'], url_data['num_special_chars'], url_data['num_numbers'], c=url_data['gmm_cluster'])
ax.set_title('EM-Clustering')
ax.set_xlabel('URL Length')
ax.set_ylabel('Number of Special Characters')
ax.set_zlabel('Number of Numbers')
plt.show()