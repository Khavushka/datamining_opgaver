'''
Exercises 5.1
'''
# Importing libraries
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans

# Tabel 1
#Define datapunkterne 
tabel1 = [[10,1],[2,3], [3,4], [1,5], [1,7], [6,8], [7,8], [7,9]]

#Vi har to forskellige cluster
kmeans_model = KMeans(n_clusters=2)

# Cluster the data and obtain the cluster 
labels = kmeans_model.fit_predict(tabel1)

silhouette_coefficient = silhouette_score(tabel1, labels)
print('Silhouette_coefficient fra tabel 1:', silhouette_coefficient)

# Svar: silhouette_coefficient: 0.49610653006282945


# Tabel 2

tabel2 = [[10,1],[2, 3], [3,4], [1,4], [7,7], [6,8], [7,8], [7,9]]

#Vi har to forskellige cluster
kmeans_model = KMeans(n_clusters=2)

# Cluster the data and obtain the cluster 
labels = kmeans_model.fit_predict(tabel2)

silhouette_coefficient = silhouette_score(tabel2, labels)
print('silhouette_coefficient fra tabel 2:', silhouette_coefficient)

# Svar: silhouette_coefficient fra tabel 2: 0.571443656751857

# Tabel 3

tabel3 = [[10,1],[2,3], [3,4], [1,5], [7,7], [6,8], [7,8], [7,9]]

#Vi har to forskellige cluster
kmeans_model = KMeans(n_clusters=2)

# Cluster the data and obtain the cluster 
labels = kmeans_model.fit_predict(tabel3)

silhouette_coefficient = silhouette_score(tabel3, labels)
print('silhouette_coefficient fra tabel 3:', silhouette_coefficient)

# Svar: silhouette_coefficient fra tabel 3: 0.5467209394773982

