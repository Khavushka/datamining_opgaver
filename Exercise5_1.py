# '''
# Exercises 5.1
# '''
# # Importing libraries

import numpy as np

# Define data points and corresponding labels
X = np.array([[10,1],[2,3], [3,4], [1,5], [7,7], [6,8], [7,8], [7,9]])
labels1 = np.array([1, 1, 1, 1, 0, 0, 0, 0])
labels2 = np.array([1, 0, 0, 0, 0, 0, 0, 0])
labels3 = np.array([0, 1, 1, 1, 0, 0, 0, 0])

# Compute TD^2 for each label array
for labels in [labels1, labels2, labels3]:
    centroids = []
    for i in np.unique(labels):
        centroids.append(np.mean(X[labels == i], axis=0))

    TD_squared = 0
    for i in range(len(X)):
        TD_squared += np.linalg.norm(X[i] - centroids[labels[i]]) ** 2

    print("The TD^2 value for labels", labels, "is:", TD_squared)