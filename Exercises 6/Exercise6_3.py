import numpy as np
from scipy.spatial.distance import cityblock
from collections import Counter

# Define the data points and their labels
X = np.array([[9, 3],[8, 2],[7,2],[7,4],[6, 1],[6,3],[6,5],[6, 6],[5,4],[5, 5],[5, 7],[4, 4],[4, 6], [2, 5], [2, 8]])
y = ['square', 'square', 'square', 'square', 'square', 'square', 'circle', 'circle', 'square', 'circle', 'circle', 'circle', 'circle', 'circle', 'circle']
# y = np.array([0,0,0,0,0,1,1,0,1,1,1,1,1,1])

# Define the test point
test_point = np.array([6, 6])

# Define the value of k
# k = 4
# k = 7
# k = 9
# k = 13
k = 15

# Beregn Manhattan-afstandene mellem testpunktet og -alle datapunkter
distances = [cityblock(test_point, x) for x in X]
# print(distances)

# Få indeksene for de k nærmeste naboer
knn_indices = np.argsort(distances)[:k]

# Få etiketterne på de k nærmeste naboer
knn_labels = [y[i] for i in knn_indices]

# Forekomsterne af hver etiket i de k nærmeste naboer
label_counts = Counter(knn_labels)

# Tildel testpunktet til majoritetsklassen blandt de k nærmeste naboer
predicted_class = label_counts.most_common(1)[0][0] 
print('The predicted class of the query point is:', predicted_class)