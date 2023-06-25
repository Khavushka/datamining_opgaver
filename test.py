import numpy as np

# Calculate the distance between two points
point1 = np.array([0, 2])
point2 = np.array([0, 0])
distance = np.linalg.norm(point1 - point2)

print("Distance between (0, 2) and (0, 0):", distance)

# Define the matrix transformation
M1 = np.array([[1, 0], [0, 4]])
# p = np.array([-np.sqrt(8), np.sqrt(2)])
p = np.array([4, 0])
origin = np.array([0, 0])

# Apply the matrix transformation
Q_M1 = np.sqrt(np.dot(np.dot((p - origin), M1), np.transpose(p - origin)))

print("Distance after transformation:", Q_M1)
