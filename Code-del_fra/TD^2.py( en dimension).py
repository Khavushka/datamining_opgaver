def compute_td_squared(data, labels):
    unique_labels = set(labels)
    td_squared = 0

    for label in unique_labels:
        cluster_data = [x for x, y in zip(data, labels) if y == label]
        cluster_mean = sum(cluster_data) / len(cluster_data)
        cluster_td_squared = sum((x - cluster_mean) ** 2 for x in cluster_data)
        td_squared += cluster_td_squared

    return td_squared

# Example usage
data = [2, 4, 6, 10, 14, 16, 18]  # One-dimensional dataset
labels = [0, 0, 0, 1, 1, 1, 1]  # Cluster labels

td_squared = compute_td_squared(data, labels)
print("Total Deviation Squared (TD^2):", td_squared)

