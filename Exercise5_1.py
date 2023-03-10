'''
 Exercises 5.1
'''
# Importing libraries
import numpy as np
from sklearn.metrics import silhouette_score

# Define data points and corresponding labels
X = np.array([[10,1],[2,3], [3,4], [1,5], [7,7], [6,8], [7,8], [7,9]])
labels1 = np.array([1, 1, 1, 1, 0, 0, 0, 0])
labels2 = np.array([1, 0, 0, 0, 0, 0, 0, 0])
labels3 = np.array([0, 1, 1, 1, 0, 0, 0, 0])

'''
Dette kodestykke beregner TD^2 værdien for hvert sæt label ved hjælp af dataene i X.
TD^2 -værdien er et mål for spredningen af datapunkterne omkring deres centroids.
Beregner summen af de kvadrerede afstande mellem hvert datapunkt og dets tildelts tyngdepunkt.
'''

# Loop går gennem hvert label og beregnerpunkterne ved hjælp af funktionen np.mean
for labels in [labels1, labels2, labels3]:
    centroids = []
    # Looper igennem hvert datapunkt i X
    for i in np.unique(labels):
        centroids.append(np.mean(X[labels == i], axis=0))
    # Beregner afstand mellem punktet og dets tildelte label og tilføjer den afstand til TD^2
    TD_squared = 0
    for i in range(len(X)):
        TD_squared += np.linalg.norm(X[i] - centroids[labels[i]]) ** 2
    # Udskriver TD^2 værdien for hvert punkt sammen med det tilsvarende sæt label
    print("The TD^2 value for labels", labels, "is:", TD_squared)

for sil in [labels1, labels2, labels3]:
    silhouette = []
    for i in sil:
        silhouette = silhouette_score(X, sil)
    print("The silhouette value for labels:", silhouette)

'''
Svar:
The TD^2 value for labels [1 1 1 1 0 0 0 0] is: 61.5
The TD^2 value for labels [1 0 0 0 0 0 0 0] is: 72.85714285714285
The TD^2 value for labels [0 1 1 1 0 0 0 0] is: 54.400000000000006
The silhouette value for labels: 0.5161518748019878
The silhouette value for labels: 0.41330904778417
The silhouette value for labels: 0.5467209394773982

'''