def calculate_gini_index(class_counts):
    total_count = sum(class_counts)
    gini_index = 1

    for count in class_counts:
        proportion = count / total_count
        gini_index -= proportion ** 2

    return gini_index

def calculate_weighted_gini_index(class_counts_list, weights_list):
    weighted_gini_index = 0

    for class_counts, weight in zip(class_counts_list, weights_list):
        gini_index = calculate_gini_index(class_counts)
        weighted_gini_index += weight * gini_index

    return weighted_gini_index

# Example usage
class_counts_list = [[0, 2], [3, 0]]  # Class counts for weak and strong winds
weights_list = [2/5, 3/5]  # Proportions of observations for weak and strong winds
gini_index = calculate_weighted_gini_index(class_counts_list, weights_list)
print("Gini Index:", gini_index)
