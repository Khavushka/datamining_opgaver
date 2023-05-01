# Decision trees, naive Bayes, and k-nn classification
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

# Split dataset into training
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
