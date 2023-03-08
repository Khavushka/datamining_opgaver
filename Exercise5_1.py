'''
Exercises 5.1
'''

from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans

# Tabel 1

#Define datapunkterne 
x = [[10,1],[2,3], [3,4], [1,5], [1,7], [6,8], [7,8], [7,9]]

#Vi har to forskellige cluster
kmeans_model = KMeans(n_clusters=2)

# Cluster the data and obtain the cluster 
labels = kmeans_model.fit_predict(x)

silhouette_coefficient = silhouette_score(x, labels)
print('silhouette_coefficient:', silhouette_coefficient)

# Svar: silhouette_coefficient: 0.49610653006282945


# Tabel 2