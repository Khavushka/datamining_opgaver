import numpy as np
from sklearn import metrics

#({A,B,C,D,E,F,G,H,I,J})
#{ 1,2,3,4,5,6,7,8,9,10}
#TALLENE svare til bokstaverne fx. A=1 
# Rankings for each method
rankings = {
    'm1': [3, 4, 1, 5, 6, 2, 7, 8, 9, 10],
    'm2': [10, 1, 4, 3, 6, 7, 2, 8, 9, 5],
    'm3': [9, 4, 1, 5, 6, 7, 2, 8, 3, 10],
    'm4': [9, 10, 5, 1, 2, 6, 7, 8, 3, 4],
    'm5': [9, 1, 5, 10, 8, 3, 4, 2, 6, 7],
}

# True labels (outliers)
outliers = [1, 2]

# Calculate ROC and AUC for each method
roc_scores = {}
auc_scores = {}

for method, ranks in rankings.items():
    roc_scores[method] = np.mean([r for r, point in zip(ranks, outliers) if point in outliers])
    y_true = [point in outliers for point in rankings['m1']]
    y_scores = [-rank for rank in ranks]  # Negate ranks for AUC calculation
    if method == 'm1':
        y_scores = [-rank for rank in rankings['m5']]  # Use m5 scores for m1 AUC calculation
    auc_scores[method] = metrics.roc_auc_score(y_true, y_scores)

# Print the ROC scores
print("ROC Scores:")
for method, score in roc_scores.items():
    print(f"{method}: {score}")

# Print the AUC scores
print("AUC Scores:")
for method, score in auc_scores.items():
    print(f"{method}: {score}")


