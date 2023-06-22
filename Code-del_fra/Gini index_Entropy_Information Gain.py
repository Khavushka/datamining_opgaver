import math

def calculate_gini_index(class_counts):
    total_count = sum(class_counts)
    gini_index = 1

    for count in class_counts:
        proportion = count / total_count
        gini_index -= proportion ** 2

    return gini_index

def calculate_entropy(class_counts):
    total_count = sum(class_counts)
    entropy = 0

    for count in class_counts:
        proportion = count / total_count
        if proportion != 0:
            entropy -= proportion * math.log2(proportion)

    return entropy

def calculate_information_gain(class_counts_list, weights_list):
    total_count = sum(sum(class_counts) for class_counts in class_counts_list)
    entropy_parent = calculate_entropy([sum(class_counts) for class_counts in class_counts_list])
    information_gain = entropy_parent

    for class_counts, weight in zip(class_counts_list, weights_list):
        weight_sum = sum(class_counts)
        weight_proportion = weight_sum / total_count
        entropy_child = calculate_entropy(class_counts)
        information_gain -= weight_proportion * entropy_child

    return information_gain

def calculate_weighted_gini_index(class_counts_list, weights_list):
    weighted_gini_index = 0

    for class_counts, weight in zip(class_counts_list, weights_list):
        gini_index = calculate_gini_index(class_counts)
        weighted_gini_index += weight * gini_index

    return weighted_gini_index

# Example usage
class_counts_list = [[1, 1], [1, 2]]  # Class counts for weak and strong winds
weights_list = [2/5, 3/5]  # Proportions of observations for weak and strong winds

gini_index = calculate_weighted_gini_index(class_counts_list, weights_list)
entropy = calculate_entropy([sum(class_counts) for class_counts in class_counts_list])
information_gain = calculate_information_gain(class_counts_list, weights_list)

print("Gini Index:", gini_index)
print("Entropy:", entropy)
print("Information Gain:", information_gain)

