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
