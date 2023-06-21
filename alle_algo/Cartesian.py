import math

# Coordinates of the points
A = (0, 0)
B = (0, 1)
C = (1, 1)

# Calculate distances using different distance measures

# L_0.8 distance measure
d_AB_08 = ((abs(A[0] - B[0]) ** 0.8) + (abs(A[1] - B[1]) ** 0.8)) ** (1 / 0.8)
d_AC_08 = ((abs(A[0] - C[0]) ** 0.8) + (abs(A[1] - C[1]) ** 0.8)) ** (1 / 0.8)

# L_1 distance measure
d_AB_1 = abs(A[0] - B[0]) + abs(A[1] - B[1])
d_AC_1 = abs(A[0] - C[0]) + abs(A[1] - C[1])

# L_2 (Euclidean) distance measure
d_AB_2 = math.sqrt((A[0] - B[0]) ** 2 + (A[1] - B[1]) ** 2)
d_AC_2 = math.sqrt((A[0] - C[0]) ** 2 + (A[1] - C[1]) ** 2)

# L_infinity (Chebyshev) distance measure
d_AB_inf = max(abs(A[0] - B[0]), abs(A[1] - B[1]))
d_AC_inf = max(abs(A[0] - C[0]), abs(A[1] - C[1]))

# Print the distances
print("Distances using different distance measures:")
print("L_0.8:")
print("Distance between A and B:", d_AB_08)
print("Distance between A and C:", d_AC_08)
print("L_1:")
print("Distance between A and B:", d_AB_1)
print("Distance between A and C:", d_AC_1)
print("L_2:")
print("Distance between A and B:", d_AB_2)
print("Distance between A and C:", d_AC_2)
print("L_infinity:")
print("Distance between A and B:", d_AB_inf)
print("Distance between A and C:", d_AC_inf)
