'''
Neutral networks and support vector machines.
- what do you observe for the training time?
- what do you observe for the apparent error and the true error?
'''

# import library
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# load dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split dataset into training og testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)