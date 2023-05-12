import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
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
# print("Decision tree accuracy:", dt_acc)
print(clf_pred)


# Random forest classifier
rf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)
rf_acc = accuracy_score(y_test, rf_pred)
# print("Random Forest accuracy:", rf_acc)

# Neural network classifier
nn = MLPClassifier(hidden_layer_sizes=(100,), random_state=42)
nn.fit(X_train, y_train)
nn_pred = nn.predict(X_test)
nn_acc = accuracy_score(y_test, nn_pred)
# print("Neural Network accuracy:", nn_acc)

print("Cancer data set  dimensions: {}".format(dataset.shape))
print(dataset.head())
results=pd.DataFrame({"Model": ["Decision Tree", "Random Forest", "Neural Network"],
    "Accuracy": [dt_acc, rf_acc, nn_acc]})
print(results)


models = ["Decision Tree", "Random Forest", "Neural Network"]
accuracy = [dt_acc, rf_acc, nn_acc]
plt.bar(models, accuracy)
plt.title("Accuracy Scores for Breast Cancer Dataset")
plt.xlabel("Model")
plt.ylabel("Accuracy")
plt.show()

fig = plt.figure(figsize=(25,20))
_ = tree.plot_tree(clf, 
                    filled=True)


# new data visualisation
plt.rcParams['font.size']=10
sns.pairplot(clf, hue='Status', palette='Blues')