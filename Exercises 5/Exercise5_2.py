'''
Exercise 5.2
'''
import numpy as np
import matplotlib.pyplot as plt

# Definere dataset
X = np.array([[10,1],[2,3], [3,4], [1,5], [7,7], [6,8], [7,8], [7,9]])
labels1 = np.array([1, 1, 1, 1, 0, 0, 0, 0])
labels2 = np.array([1, 0, 0, 0, 0, 0, 0, 0])
labels3 = np.array([0, 1, 1, 1, 0, 0, 0, 0])

# Nummer af clusters
k_values = [2, 3, 4, 5]
# k_values = [3]

for k in k_values:
    # Random centroids 
    centroids = np.array([[6.7,4], [1.5,4], [6.7,8.3], [2,2],[4,3]])
    # centroids = np.random.rand(k, 2) * 10

    # Fortsat indtil convergence
    for i in range(100):
        # Assign each data point to the closest centroid
        distances = np.sqrt(((X - centroids[:, np.newaxis])**2).sum(axis=2))
        labels = np.argmin(distances, axis=0)

        # Update centroids
        for j in range(k):
            centroids[j] = X[labels == j].mean(axis=0)
    print(centroids)
    
    # Compute TD2 
    td2 = ((X - centroids[labels])**2).sum() / X.shape[0]
    print(f'For k={k}, TD2={td2:.2f}')