import numpy as np

# Define the two points as arrays or lists
point1 = np.array([2, 8, 5])  # Replace x1, y1, z1 with the coordinates of the first point
point2 = np.array([3, 7, 6])  # Replace x2, y2, z2 with the coordinates of the second point
point3 = np.array([1, 2, 0]) # Replace x3, y3, z3

# Calculate the Euclidean distance
euclidean_distance = np.linalg.norm(point1 - point2 - point3)

# Print the Euclidean distance
print("Euclidean distance:", euclidean_distance)
