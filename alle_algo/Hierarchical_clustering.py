'''
This example demonstrates a basic usage of hierarchical clustering using SciPy. You can customize the parameters and explore different linkage methods to fit your specific requirements. Additionally, you can use other libraries like scikit-learn (sklearn.cluster.AgglomerativeClustering) for hierarchical clustering as well, providing more flexibility and options.
'''

import numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt

# Generate sample data
X = np.array([[1, 2], [3, 2], [2, 5], [4, 7], [5, 5], [6, 6]])

# Perform hierarchical clustering
Z = linkage(X, method='ward')

# Plot the dendrogram
plt.figure(figsize=(8, 6))
dendrogram(Z)
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('Data Points')
plt.ylabel('Distance')
plt.show()
