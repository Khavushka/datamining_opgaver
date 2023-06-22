# import numpy as np
# from scipy.spatial import distance

# def calculate_reachability_distance(dataset, point_index, epsilon):
#     distances = distance.cdist([dataset[point_index]], dataset)[0]
#     neighbors_indices = np.where(distances <= epsilon)[0]
#     if len(neighbors_indices) == 1:
#         return np.inf
#     else:
#         return np.max(distances[neighbors_indices[1:]])

# def find_core_points(dataset, epsilon, min_pts):
#     num_points = len(dataset)
#     core_points = []
#     for i in range(num_points):
#         reachability_dist = calculate_reachability_distance(dataset, i, epsilon)
#         if np.sum(reachability_dist <= epsilon) >= min_pts:
#             core_points.append(i)
#     return core_points

# # Define the dataset
# dataset = np.array([
#     [8, 5], [7, 5], [7, 6], [7, 7], [6, 5], [6, 6], [6, 7], [5, 4], [5, 5],
#     [5, 8], [4, 2], [4, 3], [3, 1], [3, 2], [3, 4], [3, 6], [2, 2], [2, 3], [2, 7]
# ])

# # Set the parameters
# epsilon = 2
# min_pts = 4

# # Find the core points
# core_points = find_core_points(dataset, epsilon, min_pts)

# # Check the statements
# statements = [
#     ("A", 2, 4) in core_points,
#     ("B", 2, 4) in core_points,
#     ("G", 2, 4) in core_points,
#     ("J", 2, 9) in core_points,
#     ("M", 2, 4) in core_points,
#     ("P", 2, 1) in core_points,
#     ("Q", 1, 4) in core_points,
#     ("R", 1, 4) in core_points
# ]

# # Print the results
# for i, statement in enumerate(statements, 1):
#     print(f"Statement {i}: {'Correct' if statement else 'Incorrect'}")


import numpy as np
from sklearn.metrics.pairwise import manhattan_distances
from sklearn.cluster import DBSCAN

# Define the dataset
dataset = np.array([[2, 2], [2, 3], [2, 7], [3, 1], [3, 2], [3, 4], [3, 6], [4, 2],
                    [4, 3], [5, 4], [5, 5], [5, 8], [6, 5], [6, 6], [6, 7], [7, 5],
                    [7, 6], [7, 7], [8, 5]])

# Define the labels for each point (not used in density calculations)
labels = np.arange(len(dataset))

# Define the DBSCAN parameters
epsilon = 2
min_samples = 4

# Compute pairwise Manhattan distances between all points
distances = manhattan_distances(dataset, dataset)

# Apply DBSCAN
dbscan = DBSCAN(eps=epsilon, min_samples=min_samples, metric='precomputed')
dbscan.fit(distances)

# Check the correctness of the statements

# Statement 1: P is directly density-reachable from M
is_p_reachable_from_m = dbscan.labels_[2] == dbscan.labels_[6]

# Statement 2: S is directly density-reachable from Q
is_s_reachable_from_q = dbscan.labels_[11] == dbscan.labels_[14]

# Statement 3: P and S are density-connected
is_p_density_connected_to_s = dbscan.labels_[2] == dbscan.labels_[11]

# Statement 4: A and L are density-connected
is_a_density_connected_to_l = dbscan.labels_[3] == dbscan.labels_[18]

# Print the results
print("Statement 1:", is_p_reachable_from_m)
print("Statement 2:", is_s_reachable_from_q)
print("Statement 3:", is_p_density_connected_to_s)
print("Statement 4:", is_a_density_connected_to_l)
