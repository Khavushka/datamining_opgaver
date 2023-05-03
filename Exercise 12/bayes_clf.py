# Bayes classifier
# https://soumenatta.medium.com/exploring-the-naive-bayes-classifier-algorithm-with-iris-dataset-in-python-372f5a107120

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train the Naive Bayes classifier
model = GaussianNB()
model.fit(X_train, y_train)

# Evaluate the model on the test set
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy:', accuracy)

# Make predictions on new samples
sample = [[5.1, 3.5, 1.4, 0.2]]
prediction = model.predict(sample)
print('Prediction:', prediction)