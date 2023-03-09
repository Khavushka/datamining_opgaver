import numpy as np
import matplotlib.pyplot as plt

# Define the data set
data = np.array([[1.5, 4], [6.6, 8.3], [6.6, 4]])

# Define the initial centroids
centroids = np.array([[1.5, 4], [6.6, 8.3], [6.6, 4]])

# Define the number of iterations
num_iters = 10

# Define the number of clusters
k = 2

# Define the least squares assignment function
def least_squares_assignment(data, centroids):
    distances = np.sqrt(((data - centroids[:, np.newaxis])**2).sum(axis=2))
    return np.argmin(distances, axis=0)

# Initialize a list to store the TD2 values for each iteration
td2_values = []

# Perform k-means algorithm for num_iters iterations
for i in range(num_iters):
    # Assign each data point to the closest centroid
    assignments = least_squares_assignment(data, centroids)

    # Update the centroids as the mean of the data points in each cluster
    for j in range(k):
        centroids[j] = np.mean(data[assignments == j], axis=0)

    # Compute the TD2 value
    td2 = np.sum((data - centroids[assignments])**2)
    td2_values.append(td2)

    # Plot the current state of the clustering
    colors = ['r', 'b', 'g']
    plt.figure()
    for j in range(k):
        plt.scatter(data[assignments == j, 0], data[assignments == j, 1], c=colors[j], label='Cluster {}'.format(j+1))
    plt.scatter(centroids[:, 0], centroids[:, 1], c='k', marker='x', s=100, label='Centroids')
    plt.legend()
    plt.title('Iteration {}'.format(i+1))
    # plt.show()

# Print the final TD2 value
print('Final TD2 value:', td2_values[-1])