# Step 1: Calculate the posterior probabilities of the classifiers
pr_h1_given_d = 0.5
pr_h2_given_d = 0.3
pr_h3_given_d = 0.2

# Step 2: Calculate the class probabilities given each classifier's prediction
pr_plus_given_h1 = 0.6
pr_plus_given_h2 = 0.2
pr_plus_given_h3 = 0.9
pr_minus_given_h1 = 0.4
pr_minus_given_h2 = 0.8
pr_minus_given_h3 = 0.1

# Step 3: Calculate the overall class probabilities
pr_plus = pr_h1_given_d * pr_plus_given_h1 + pr_h2_given_d * pr_plus_given_h2 + pr_h3_given_d * pr_plus_given_h3
pr_minus2 = pr_h1_given_d * pr_minus_given_h1 + pr_h2_given_d * pr_minus_given_h2 + pr_h3_given_d * pr_minus_given_h3

# Print the class probabilities
print("Pr(+):", pr_plus)
print("Pr(-2):", pr_minus2)
