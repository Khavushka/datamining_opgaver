'''
Random Variables and Probability Distributions. We played a lot with dice in the lecture.  
When we take the sum ofndice (a random variable), we get a prob-ability distribution over the possible values. 
For just one die, this distribution is discrete with equal probabilitiesover{1,2,3,4,5,6}. 
For two dies, the probabilities are unequally distributed over{2,3,4,5,6,7,8,9,10,11,12}.
How does the shape of the probability distribution develop with increasing n?
'''
# 1 solution
# import random
# import matplotlib.pyplot as plt

# def roll_dice(n):
#     return sum(random.randint(1,6) for _ in range(n))

# def simulate_rolls(n, num_trials):
#     results = [roll_dice(n) for _ in range(num_trials)]
#     distribution = [results.count(i) / num_trials for i in range(n, 6*n+1)]
#     return distribution

# # Simulerer rolling 1, 2, 3, 4 dice
# distributions = [simulate_rolls(n, 10000) for n in range(1, 5)]

# fig, axs = plt.subplots(2, 2)
# axs = axs.ravel()
# for i, dist in enumerate(distributions):
#     axs[i].plot(range(len(dist)), dist)
#     axs[i].set_title(f"Sum of {i+1} dice")
# plt.show()

# 2 solution
import numpy as np
import matplotlib.pyplot as plt

# Define the number of dice to roll
n = 5

# Generate all possible outcomes for n dice
outcomes = np.arange(n, 6*n+1)

# Calculate the probability of each outcome
probabilities = np.zeros_like(outcomes, dtype=float)
for i in range(n, 6*n+1):
    count = 0
    for j in range(1, 7):
        if i-j <= n-1:
            count += 1
    probabilities[i-n] = count / 6**n

# Plot the probability distribution
plt.bar(outcomes, probabilities)
plt.xlabel('Sum of n Dice')
plt.ylabel('Probability')
plt.title('Probability Distribution for Sum of {} Dice'.format(n))
plt.show()