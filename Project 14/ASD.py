import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

# Load dataset
df = pd.read_csv("Project 14\ASD.csv")
df = df.replace({'?': np.nan}).dropna()
print(df.head())
# Split the dataset into training and testing sets
X = df.drop(columns=["Class/ASD"])
y = df["Class/ASD"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train and evaluate Decision Tree Classifier
dtc = DecisionTreeClassifier()
# dtc.fit(X_train, y_train)
# y_pred = dtc.predict(X_test)
# dtc_acc = accuracy_score(y_test, y_pred)
# # print(f"Decision Tree Classifier Accuracy: {dtc_acc:.3f}")

# # Train and evaluate Random Forest Classifier
# rfc = RandomForestClassifier()
# rfc.fit(X_train, y_train)
# y_pred = rfc.predict(X_test)
# rfc_acc = accuracy_score(y_test, y_pred)
# print(f"Random Forest Classifier Accuracy: {rfc_acc:.3f}")

# # Train and evaluate Neural Network Classifier
# nnc = MLPClassifier(hidden_layer_sizes=(50, 50), max_iter=1000)
# nnc.fit(X_train, y_train)
# y_pred = nnc.predict(X_test)
# nnc_acc = accuracy_score(y_test, y_pred)
# print(f"Neural Network Classifier Accuracy: {nnc_acc:.3f}")
