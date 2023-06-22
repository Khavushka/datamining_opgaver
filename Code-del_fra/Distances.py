import math
import numpy as np

# def euclidean_distance(point1, point2):
#     # Ensure both points have the same dimensionality
#     if len(point1) != len(point2):
#         raise ValueError("Points must have the same dimensionality.")
    
#     # Compute the squared distance for each dimension
#     squared_distance = sum((a - b) ** 2 for a, b in zip(point1, point2))
    
#     # Take the square root of the sum of squared distances
#     distance = math.sqrt(squared_distance)
    
#     return distance

# # Example usage
# point_a = [1, 2]
# point_b = [4, 5]
# distance = euclidean_distance(point_a, point_b)
# print(f"The Euclidean distance between {point_a} and {point_b} is: {distance}")

# def manhattan_distance(point1, point2):
#     # Ensure both points have the same dimensionality
#     if len(point1) != len(point2):
#         raise ValueError("Points must have the same dimensionality.")
    
#     # Compute the sum of absolute differences for each dimension
#     distance = sum(abs(a - b) for a, b in zip(point1, point2))
    
#     return distance

# # Example usage
# point_a = [1, 2]
# point_b = [4, 5]
# distance = manhattan_distance(point_a, point_b)
# print(f"The Manhattan distance between {point_a} and {point_b} is: {distance}")

# def chebyshev_distance(point1, point2):
#     # Ensure both points have the same dimensionality
#     if len(point1) != len(point2):
#         raise ValueError("Points must have the same dimensionality.")
    
#     # Compute the maximum absolute difference among dimensions
#     distance = max(abs(a - b) for a, b in zip(point1, point2))
    
#     return distance

# # Example usage
# point_a = [1, 2]
# point_b = [4, 5]
# distance = chebyshev_distance(point_a, point_b)
# print(f"The Chebyshev distance between {point_a} and {point_b} is: {distance}")

# def weighted_euclidean_distance(point1, point2, weights):
#     # Ensure both points and weights have the same dimensionality
#     if len(point1) != len(point2) or len(point1) != len(weights):
#         raise ValueError("Points and weights must have the same dimensionality.")
    
#     # Compute the squared distance for each dimension, weighted by the corresponding weight
#     squared_distance = sum((w * (a - b) ** 2) for a, b, w in zip(point1, point2, weights))
    
#     # Take the square root of the sum of weighted squared distances
#     distance = math.sqrt(squared_distance)
    
#     return distance

# # Example usage
# point_a = [1, 2]
# point_b = [4, 5]
# weights = [0.5, 1]
# distance = weighted_euclidean_distance(point_a, point_b, weights)
# print(f"The weighted Euclidean distance between {point_a} and {point_b} is: {distance}")

def quadratic_form(matrix, vector):
    # Ensure the matrix and vector have compatible dimensions
    if matrix.shape[1] != vector.shape[0]:
        raise ValueError("Matrix and vector dimensions are not compatible.")
    
    # Calculate the quadratic form
    result = np.dot(vector.T, np.dot(matrix, vector))
    
    return result

# Example usage
matrix = np.array([[1, 2], [3, 4]])
vector = np.array([1, 2])

result = quadratic_form(matrix, vector)
print("The quadratic form result is:", result)




