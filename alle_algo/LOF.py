import numpy as np
from sklearn.neighbors import LocalOutlierFactor

# Generate sample data
X = np.array([[1, 1], [1, 2], [2, 1], [2, 2], [3, 5], [3, 9], [3, 10], [4, 10], [4,11],[5,10],[7,10],[9,4],[9,5],[10,3],[10,4],[10,5],[10,6],[10,9],[11,4],[11,5]])

# Create LOF outlier detector object
lof = LocalOutlierFactor(n_neighbors=2)

# Fit the model and predict outliers
outlier_labels = lof.fit_predict(X)

# Retrieve outlier scores
outlier_scores = lof.negative_outlier_factor_

# Print the outlier labels and scores
for i, label in enumerate(outlier_labels):
    print("Data Point:", X[i], "Outlier Label:", label, "Outlier Score:", outlier_scores[i])
