import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load the data into a pandas dataframe
data = pd.read_csv('wdbc.csv', header=None)

# Split the data into features (X) and labels (y)
X = data.iloc[:, 2:].values
y = data.iloc[:, 1].values

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Create and train a decision tree classifier
dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)
dt_pred = dt.predict(X_test)
dt_acc = accuracy_score(y_test, dt_pred)
print("Decision Tree accuracy:", dt_acc)

# Create and train a random forest classifier
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)
rf_acc = accuracy_score(y_test, rf_pred)
print("Random Forest accuracy:", rf_acc)

# Create and train a neural network classifier
nn = MLPClassifier(hidden_layer_sizes=(100,), random_state=42)
nn.fit(X_train, y_train)
nn_pred = nn.predict(X_test)
nn_acc = accuracy_score(y_test, nn_pred)
print("Neural Network accuracy:", nn_acc)
