import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
from sklearn import tree
import seaborn as sns; sns.set(style="ticks", color_codes=True)

data = load_breast_cancer()
dataset = pd.DataFrame(data=data['data'], columns=data['feature_names'])
# print(dataset)

X = dataset.copy()
y = data['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# Decision Tree Classifier
clf = DecisionTreeClassifier(ccp_alpha=0.01)
clf = clf.fit(X_train, y_train)
clf.get_params()
clf_pred = clf.predict(X_test)
clf.predict_proba(X_test)
dt_acc = accuracy_score(y_test, clf_pred)

# y_pred_dt = clf_dt.predict(X_test)
acc_dt = accuracy_score(y_test, clf_pred)
prec_dt = precision_score(y_test, clf_pred)
rec_dt = recall_score(y_test, clf_pred)
f1_dt = f1_score(y_test, clf_pred)
# print("Decision tree accuracy:", dt_acc)
# print(clf_pred)

# Vusualization decision tree
# fig, ax = plt.subplots(figsize=(10, 10))
# plot_tree(clf, ax=ax, feature_names=data.feature_names, class_names=data.target_names, filled=True)
# plt.title("Decision Tree for Breast Cancer Dataset")
# plt.xlabel("Model")
# plt.ylabel("Accuracy")
# plt.show()


# Random forest classifier
rf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)
rf_acc = accuracy_score(y_test, rf_pred)
# print("Random Forest accuracy:", rf_acc)
acc_rf = accuracy_score(y_test, rf_pred)
prec_rf = precision_score(y_test, rf_pred)
rec_rf = recall_score(y_test, rf_pred)
f1_rf = f1_score(y_test, rf_pred)



# Neural network classifier
nn = MLPClassifier(hidden_layer_sizes=(100,), random_state=42)
nn.fit(X_train, y_train)
nn_pred = nn.predict(X_test)
# nn_acc = accuracy_score(y_test, nn_pred)
# # print("Neural Network accuracy:", nn_acc)

# print("Cancer data set  dimensions: {}".format(dataset.shape))
# print(dataset.head())
# results=pd.DataFrame({"Model": ["Decision Tree", "Random Forest", "Neural Network"],
#     "Accuracy": [dt_acc, rf_acc, nn_acc]})
# print(results)

acc_nn = accuracy_score(y_test, nn_pred)
prec_nn = precision_score(y_test, nn_pred)
rec_nn = recall_score(y_test, nn_pred)
f1_nn = f1_score(y_test, nn_pred)

# Results
# models = ["Decision Tree", "Random Forest", "Neural Network"]
# accuracy = [dt_acc, rf_acc, nn_acc]
# plt.bar(models, accuracy)
# plt.title("Accuracy Scores for Breast Cancer Dataset")
# plt.xlabel("Model")
# plt.ylabel("Accuracy")
# plt.show()

print("Decision Tree Classifier Results:")
print("Accuracy: {:.4f}".format(acc_dt))
print("Precision: {:.4f}".format(prec_dt))
print("Recall: {:.4f}".format(rec_dt))
print("F1 Score: {:.4f}".format(f1_dt))
print()
print("Random Forest Classifier Results:")
print("Accuracy: {:.4f}".format(acc_rf))
print("Precision: {:.4f}".format(prec_rf))
print("Recall: {:.4f}".format(rec_rf))
print("F1 Score: {:.4f}".format(f1_rf))
print()
print("Neural Network Classifier Results:")
print("Accuracy: {:.4f}".format(acc_nn))
print("Precision: {:.4f}".format(prec_nn))
print("Recall: {:.4f}".format(rec_nn))
print("F1 Score: {:.4f}".format(f1_nn))


# fig = plt.figure(figsize=(25,20))
# _ = tree.plot_tree(clf, 
#                     filled=True)


# new data visualisation
plt.rcParams['font.size']=10
sns.pairplot(data, hue='Status', palette='Blues')