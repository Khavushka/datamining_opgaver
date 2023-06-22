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
