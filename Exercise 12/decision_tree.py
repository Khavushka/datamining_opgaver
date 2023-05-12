# import pandas as pd
# from sklearn.datasets import load_breast_cancer
# from sklearn.tree import DecisionTreeClassifier, plot_tree
# import matplotlib.pyplot as plt

# # Load the breast cancer dataset
# data = load_breast_cancer()
# X = pd.DataFrame(data.data, columns=data.feature_names)
# y = data.target

# # Train a decision tree model
# clf = DecisionTreeClassifier(random_state=42)
# clf.fit(X, y)

# # Visualize the decision tree
# fig, ax = plt.subplots(figsize=(20, 10))
# plot_tree(clf, ax=ax, feature_names=data.feature_names, class_names=data.target_names, filled=True)
# plt.show()


import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Load the breast cancer dataset
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

# Split data into training og testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train and evaluate a decision tree classifier
clf_dt = DecisionTreeClassifier(ccp_alpha=0.01)
clf_dt.fit(X_train, y_train)
y_pred_dt = clf_dt.predict(X_test)

accuracy_dt = accuracy_score(y_test, y_pred_dt)
precision_dt = precision_score(y_test, y_pred_dt)
recall_dt = recall_score(y_test, y_pred_dt)
f1_dt = f1_score(y_test, y_pred_dt)
cm_dt = confusion_matrix(y_test, y_pred_dt)

# Train and evaluate a random forest classifier
clf_rf = RandomForestClassifier(random_state=42)
clf_rf.fit(X_train, y_train)
y_pred_rf = clf_rf.predict(X_test)

accuracy_rf = accuracy_score(y_test, y_pred_rf)
precision_rf = precision_score(y_test, y_pred_rf)
recall_rf = recall_score(y_test, y_pred_rf)
f1_rf = f1_score(y_test, y_pred_rf)
cm_rf = confusion_matrix(y_test, y_pred_rf)

# Train and evaluate a neural network classifier
clf_nn = MLPClassifier(random_state=42, max_iter=500)
clf_nn.fit(X_train, y_train)
y_pred_nn = clf_nn.predict(X_test)

accuracy_nn = accuracy_score(y_test, y_pred_nn)
precision_nn = precision_score(y_test, y_pred_nn)
recall_nn = recall_score(y_test, y_pred_nn)
f1_nn = f1_score(y_test, y_pred_nn)
cm_nn = confusion_matrix(y_test, y_pred_nn)

# Print the performance metrics for each classifier
print("Decision Tree Classification Results:")
print(f"Accuracy: {accuracy_dt:.3f}")
print(f"Precision: {precision_dt:.3f}")
print(f"Recall: {recall_dt:.3f}")
print(f"F1-score: {f1_dt:.3f}")
print(f"Confusion Matrix:\n{cm_dt}\n")

print("Random Forest Classification Results:")
print(f"Accuracy: {accuracy_rf:.3f}")
print(f"Precision: {precision_rf:.3f}")
print(f"Recall: {recall_rf:.3f}")
print(f"F1-score: {f1_rf:.3f}")
print(f"Confusion Matrix:\n{cm_rf}\n")

print("Neural Network Classification Results:")
print(f"Accuracy: {accuracy_nn:.3f}")
print(f"Precision: {precision_nn:.3f}")
print(f"Recall: {recall_nn:.3f}")
print(f"F1-score: {f1_nn:.3f}")
print(f"Confusion Matrix:\n{cm_nn}")







# # dataset = dataset.drop(["Unnamed: 32"], axis = 1)
# M = dataset[dataset.diagnosis == "M"]
# B = dataset[dataset.diagnosis == "B"]
# dataset.diagnosis = [1 if i == "M" else 0 for i in dataset.diagnosis]

