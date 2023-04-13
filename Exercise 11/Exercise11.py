import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN

# load the dataset
df = pd.read_csv('data.txt', sep='\t', header=None)
# plt.scatter(df[0], df[1], s=5)
# # plt.show()

# # implement K-means clustering algorithm on the dataset
# kmeans = KMeans(n_clusters=2, random_state=0)
# labels = kmeans.fit_predict(df)
# plt.scatter(df[0], df[1], c=labels, s=5)
# # plt.show()

# # implement DBSCAN clustering 
# dbscan = DBSCAN(eps=0.2, min_samples=10)
# labels = dbscan.fit_predict(df)
# plt.scatter(df[0], df[1], c=labels, s=5)
# # plt.show()

# # implement Agglomerative clustering
agglo = AgglomerativeClustering(n_clusters=2)
labels = agglo.fit_predict(df)
plt.scatter(df[0], df[1], c=labels, s=5)
plt.show()

# algorithms for he dataset

# algorithm of quality changing params

# measures fit io the structure in the dataset

# measures a systematic preference for aome of the algorithms
