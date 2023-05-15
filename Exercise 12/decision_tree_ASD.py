import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load the dataset into a pandas DataFrame
df = pd.read_csv('Project 14\ASD_new.csv')

# Split the dataset into training and testing sets
X = df.drop('target_column', axis=1) # select all columns except the target column
y = df['target_column'] # select only the target column
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Create a decision tree classifier and fit it to the training data
clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)

# Make predictions on the testing data and evaluate the accuracy
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy:', accuracy)
