import pyfpgrowth

# Transactional dataset
transactions = [
    ['apple', 'banana', 'cherry'],
    ['apple', 'banana'],
    ['banana', 'cherry'],
    ['apple', 'banana'],
    ['apple', 'cherry']
]

# Minimum support threshold
min_support = 2

# Find frequent patterns
patterns = pyfpgrowth.find_frequent_patterns(transactions, min_support)

# Find closest frequency patterns
closest_frequency_patterns = {}
min_difference = float('inf')

for pattern, support in patterns.items():
    difference = abs(support - min_support)
    if difference < min_difference:
        min_difference = difference
        closest_frequency_patterns = {pattern: support}
    elif difference == min_difference:
        closest_frequency_patterns[pattern] = support

# Print the closest frequency patterns
for pattern, support in closest_frequency_patterns.items():
    print(f"Pattern: {pattern}, Support: {support}")


'''
The same but with ABCD and support 4
'''

# import pyfpgrowth

# # Transactional dataset
# transactions = [
#     ['A', 'B', 'C'],
#     ['A', 'B'],
#     ['B', 'C', 'D'],
#     ['A', 'B'],
#     ['A', 'C', 'D', 'E']
# ]

# # Minimum support threshold
# min_support = 4

# # Find frequent patterns
# patterns = pyfpgrowth.find_frequent_patterns(transactions, min_support)

# # Find closed patterns
# closed_patterns = {}
# for pattern, support in patterns.items():
#     is_closed = True
#     for other_pattern, other_support in patterns.items():
#         if pattern != other_pattern and support == other_support and set(pattern).issubset(set(other_pattern)):
#             is_closed = False
#             break
#     if is_closed:
#         closed_patterns[pattern] = support

# # Print the closed patterns
# for pattern, support in closed_patterns.items():
#     print(f"Pattern: {pattern}, Support: {support}")


'''
# Consider the following set of frequent 3-itemsets: {1, 2, 3}, {1, 2, 4}, {1, 2, 5}, {1, 3, 4}, {1, 3, 5}, {2, 3, 4}, {2, 3, 5}, {3, 4, 5}
# assume that there are only five items in the data set 
from itertools import combinations

# Set of frequent 3-itemsets
frequent_itemsets = [
    {1, 2, 3},
    {1, 2, 4},
    {1, 2, 5},
    {1, 3, 4},
    {1, 3, 5},
    {2, 3, 4},
    {2, 3, 5},
    {3, 4, 5}
]

# List of all possible 2-item combinations
all_combinations = list(combinations(range(1, 6), 2))

# Find the closed frequent itemsets
closed_itemsets = []
for itemset in frequent_itemsets:
    is_closed = True
    for other_itemset in frequent_itemsets:
        if itemset != other_itemset and itemset.issubset(other_itemset):
            is_closed = False
            break
    if is_closed:
        closed_itemsets.append(itemset)

# Find the support count for each closed itemset
support_counts = {}
for itemset in closed_itemsets:
    support_counts[itemset] = sum(1 for transaction in frequent_itemsets if itemset.issubset(transaction))

# Print the closed frequent itemsets with their support counts
for itemset, support in support_counts.items():
    print(f"Itemset: {itemset}, Support: {support}")

'''