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
print(iris)

# Split dataset into training
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=100)

# Decision Tree Classifier
# tree_klas = DecisionTreeClassifier(random_state=42)
# tree_klas.fit(X_train, y_train)
# print(tree_klas)

clf_entropy = DecisionTreeClassifier(
    criterion = "entropy",
    random_state = 100,
    max_depth=3,
    min_samples_leaf=5,
)

clf_entropy.fit(X_train, y_train)

# Naive Bayes Classifier

# k-NN Classifier
