# Decision trees, naive Bayes, and k-nn classification
# https://www.simplilearn.com/tutorials/machine-learning-tutorial/decision-tree-in-python
import numpy as np
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Henter Iris-dataset
iris = load_iris()
X = iris.data
y = iris.target
# print(iris)

# Split dataset into training
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=100)

# Decision Tree Classifier
tree_clf = DecisionTreeClassifier(random_state=42)
tree_clf.fit(X_train, y_train)
tree_pred = tree_clf.predict(X_test)
tree_acc = accuracy_score(y_test, tree_pred)
print("Decision Tree:", tree_acc)

# Naive Bayes Classifier
nb_clf = GaussianNB()
nb_clf.fit(X_train, y_train)
nb_pred = nb_clf.predict(X_test)
nb_acc = accuracy_score(y_test, nb_pred)
print("Naive Bayes:", nb_acc)

# k-NN Classifier
knn_clf = KNeighborsClassifier(n_neighbors=3)
knn_clf.fit(X_train, y_train)
knn_pred = knn_clf.predict(X_test)
knn_acc = accuracy_score(y_test, knn_pred)
print("k-NN:", knn_acc)


# Der her er ikke noget til opgaven, mere til at afprøve noget ting
clf_entropy = DecisionTreeClassifier(
    criterion = "entropy",
    random_state = 100,
    max_depth=3,
    min_samples_leaf=5,
)
clf_entropy.fit(X_train, y_train)