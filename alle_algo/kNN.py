# Given the following dataset and Manhattan distance as distance function, not counting the query object in its neighborhood, which of the subsets are in correct decreasing order w.r.t. kNN outlier score with k=2?

import math

def manhattan_distance(point1, point2):
    return abs(point1[0] - point2[0]) + abs(point1[1] - point2[1])

def kNN_outlier_score(data, k):
    outlier_scores = []

    for query_point in data:
        distances = []
        for point in data:
            if point != query_point:
                distances.append(manhattan_distance(query_point, point))
        
        distances.sort()
        k_nearest_distances = distances[:k]
        outlier_scores.append(sum(k_nearest_distances) / k)
    
    return outlier_scores

# Dataset
data = [(5, 2), (6, 6), (5, 5), (4, 6), (4, 7), (3, 4), (3, 7), (1, 7)]

k = 2

outlier_scores = kNN_outlier_score(data, k)

# Sort the points based on outlier scores in decreasing order
sorted_data = [point for _, point in sorted(zip(outlier_scores, data), reverse=True)]

# Given options
option1 = [(5, 2), (1, 7), (6, 6)]
option2 = [(3, 4), (6, 6), (4, 7)]
option3 = [(3, 4), (4, 7), (4, 6)]

# Check if the sorted_data matches any of the given options
if sorted_data == option1:
    print("Option 1 is correct.")
elif sorted_data == option2:
    print("Option 2 is correct.")
elif sorted_data == option3:
    print("Option 3 is correct.")
else:
    print("None of the given options is correct.")


