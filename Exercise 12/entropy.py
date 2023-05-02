from math import log2

def entropy(probabilities):
    return -sum(p * log2(p) for p in probabilities if p != 0)

# Example usage
probabilities = [3/3, 0/3]
print(entropy(probabilities))