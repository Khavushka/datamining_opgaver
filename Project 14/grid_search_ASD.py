from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, make_scorer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split
import pandas as pd

# Load the dataset
data = pd.read_csv('Project 14\ASD_new.csv')

# Split the data into training, validation, and test sets
X_train, X_test, y_train, y_test = train_test_split(data.drop('Class/ASD', axis=1), data['Class/ASD'], test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=42)

# Define the hyperparameter space for the decision tree classifier
param_grid = {'max_depth': [3, 5, 7, 9],
              'min_samples_split': [2, 4, 6, 8],
              'min_samples_leaf': [1, 2, 3, 4],
              'criterion': ['gini', 'entropy']}

# Create a decision tree classifier
dt = DecisionTreeClassifier()

# Create a grid search object
grid_search = GridSearchCV(dt, param_grid, scoring=make_scorer(accuracy_score), cv=5)

# Fit the grid search object to the training data
grid_search.fit(X_train, y_train)

# Get the best hyperparameters and the best score
best_params = grid_search.best_params_
best_score = grid_search.best_score_

# Evaluate the best model on the test data
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)
test_score = accuracy_score(y_test, y_pred)

# Create a decision tree classifier
dt = DecisionTreeClassifier(max_depth=5, min_samples_split=2, min_samples_leaf=1, criterion='gini')

# Fit the classifier to the training data
dt.fit(X_train, y_train)

# Make predictions on the test data
y_pred = dt.predict(X_test)

# Evaluate the performance using the accuracy score
acc = accuracy_score(y_test, y_pred)
print("Accuracy: {:.2f}%".format(acc * 100))
