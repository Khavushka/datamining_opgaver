import matplotlib.pyplot as plt

point1 = (0, 0)
point2 = (2, 0)
point3 = (0, 2)
point4 = (2, 2)

points = [point1, point2, point3, point4]
labels = ['A', 'B', 'B', 'A']

for point, label in zip(points, labels):
    plt.scatter(point[0], point[1], label=label)

plt.legend()
plt.show()