import math

def calculate_distance(point1, point2):
    x1, y1 = point1
    x2, y2 = point2
    return math.sqrt((x1 - y1) ** 2 + (x2 - y2) ** 2)

# Define the given points
points = [
    (4, 0), 
    (math.sqrt(8), 0), 
    (2, 0), 
    (math.sqrt(2), 0), 
    (1, 0), 
    (0, -2)]
# points = [(5, 6), (math.sqrt(8), 10), (12, 3), (math.sqrt(2), 0), (1, 0), (0, -2)]

# Calculate the distance between (0,0) and (0,2)
origin = (0, 0)
target_point = (0, 2)
target_distance = calculate_distance(origin, target_point)

# Check which points have the same distance as (0,2)
same_distance_points = []
for point in points:
    distance = calculate_distance(origin, point)
    if distance == target_distance:
        same_distance_points.append(point)

# Print the points with the same distance
for point in same_distance_points:
    print(point)
