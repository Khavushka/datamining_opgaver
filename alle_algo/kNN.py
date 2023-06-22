# Given the following dataset and Manhattan distance as distance function, not counting the query object in its neighborhood, which of the subsets are in correct decreasing order w.r.t. kNN outlier score with k=2?

def manhattan_distance(x1, x2):
    return abs(x1[0] - x2[0]) + abs(x1[1] - x2[1])

def kNN_outlier_score(dataset, query, k):
    distances = [manhattan_distance(query, data) for data in dataset]
    sorted_indices = sorted(range(len(distances)), key=lambda x: distances[x])
    k_nearest_distances = distances[1:k+1]  # Exclude the query object itself
    average_distance = sum(k_nearest_distances) / k
    kth_nearest_distance = distances[k]
    outlier_score = kth_nearest_distance - average_distance
    return outlier_score

# Define the dataset
dataset = [
    [1, 7],  # B
    [3, 7],  # H
    [3, 4],  # C
    [4, 7],  # E
    [4, 6],  # G
    [5, 5],  # F
    [5, 2],  # A
    [6, 6]   # D
]

# Define the query object
query = [5, 6]

# Calculate the outlier scores for each option
option1 = [kNN_outlier_score(dataset, dataset[1], 2),
           kNN_outlier_score(dataset, dataset[7], 2),
           kNN_outlier_score(dataset, dataset[0], 2)]

option2 = [kNN_outlier_score(dataset, dataset[5], 2),
           kNN_outlier_score(dataset, dataset[0], 2),
           kNN_outlier_score(dataset, dataset[4], 2)]

option3 = [kNN_outlier_score(dataset, dataset[5], 2),
           kNN_outlier_score(dataset, dataset[4], 2),
           kNN_outlier_score(dataset, dataset[3], 2)]

# Check the order of the outlier scores
if sorted(option1, reverse=True) == option1:
    print("Option 1 is in correct decreasing order.")

if sorted(option2, reverse=True) == option2:
    print("Option 2 is in correct decreasing order.")

if sorted(option3, reverse=True) == option3:
    print("Option 3 is in correct decreasing order.")
