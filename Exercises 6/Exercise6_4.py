import math

# define the set of points
points = {
    'A': (0, 0),
    'B': (0, 1),
    'C': (1, 0),
    'D': (1, 1)
}

# Calculate the Euclidean distance between two points
def euclidean_distance(p1, p2):
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

# The nearest neighbor classifier function
def nearest_neighbor_classifier(query_point):
    # initialize the nearest neighbor and its distance to infinity
    nearest_neighbor = None
    nearest_neighbor_distance = math.inf
    
    # iterate all the training examples
    for label, point in points.items():
        # if the current point is the same as the query point, skip it
        if point == query_point:
            continue
        # print(query_point)
            
        # calculate the distance between the query point and the current training example
        distance = euclidean_distance(query_point, point)
        # print(distance, point)
        # print('----------------')
        
        # update the nearest neighbor and its distance if the current example is closer than the previous nearest neighbor
        if distance < nearest_neighbor_distance:
            nearest_neighbor = label
            nearest_neighbor_distance = distance
            # print(distance, point)
    
    # return the label of the nearest neighbor
    return nearest_neighbor

# test the classifier with all the points as query points
for label, point in points.items():
    predicted_label = nearest_neighbor_classifier(point)
    if predicted_label != label:
        print(f"Incorrect classification for query point {label}. Predicted label: {predicted_label},{point}")