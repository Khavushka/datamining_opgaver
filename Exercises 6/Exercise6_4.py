import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

point1 = (0, 0)
point2 = (2, 0)
point3 = (0, 2)
point4 = (2, 2)

points = [point1, point2, point3, point4]
labels = ['A', 'B', 'B', 'A']

for point, label in zip(points, labels):
    plt.scatter(point[0], point[1], label=label)

# plt.legend()
# plt.show()

X = points
y = labels

clf = KNeighborsClassifier(n_neighbors=1)
clf.fit(X, y)

for i, point in enumerate(points):
    query_point = point
    true_label = labels[i]
    predicted_label = clf.predict([query_point])[0]
    if true_label != predicted_label:
        print(f"Query point {query_point} has true label {true_label} but was predicted as {predicted_label}")