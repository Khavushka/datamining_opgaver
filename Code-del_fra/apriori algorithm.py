from itertools import combinations

def generate_candidates(itemsets, k):
    candidates = set()
    for itemset1 in itemsets:
        for itemset2 in itemsets:
            union = itemset1.union(itemset2)
            if len(union) == k:
                candidates.add(frozenset(union))
    return candidates

def prune(itemsets, candidates, min_support):
    pruned_candidates = set()
    item_counts = {}
    for itemset in itemsets:
        for candidate in candidates:
            if candidate.issubset(itemset):
                item_counts[candidate] = item_counts.get(candidate, 0) + 1

    for candidate, count in item_counts.items():
        support = count / len(itemsets)
        if support >= min_support:
            pruned_candidates.add(candidate)

    return pruned_candidates

def apriori(transactions, min_support):
    itemsets = [frozenset([item]) for transaction in transactions for item in transaction]
    frequent_itemsets = []
    k = 2

    while itemsets:
        candidates = generate_candidates(itemsets, k)
        candidates = prune(transactions, candidates, min_support)
        frequent_itemsets.extend(candidates)
        itemsets = candidates
        k += 1

    return frequent_itemsets

# Example usage:
transactions = [
    ['A', 'B', 'C', 'D'],
    ['B', 'C', 'E'],
    ['A', 'B', 'C', 'E'],
    ['B', 'D', 'E'],
]

min_support = 0.05

frequent_itemsets = apriori(transactions, min_support)
print("Frequent Itemsets:")
for itemset in frequent_itemsets:
    print(list(itemset))
