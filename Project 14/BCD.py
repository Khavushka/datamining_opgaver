import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
from sklearn import tree
import seaborn as sns; sns.set(style="ticks", color_codes=True)

# Henter data
data = load_breast_cancer()
dataset = pd.DataFrame(data=data.data, columns = data.feature_names)
# dataset['diagnosis'].value_counts()
# dataset = pd.DataFrame(data=data, columns=data['feature_names'])
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target
# y = data['target']


# Split data into training og testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

# Decision Tree Classifier
clf_dt = DecisionTreeClassifier(ccp_alpha=0.01)
clf_dt.fit(X_train, y_train)
y_pred_dt = clf_dt.predict(X_test)

accuracy_dt = accuracy_score(y_test, y_pred_dt)
precision_dt = precision_score(y_test, y_pred_dt)
recall_dt = recall_score(y_test, y_pred_dt)
f1_dt = f1_score(y_test, y_pred_dt)
cm_dt = confusion_matrix(y_test, y_pred_dt)

# Vusualization decision tree
fig, ax = plt.subplots(figsize=(10, 10))
plot_tree(clf_dt, ax=ax, feature_names=data.feature_names, class_names=data.target_names, filled=True)
plt.title("Decision Tree for Breast Cancer Dataset")
plt.xlabel("Model")
plt.ylabel("Accuracy")
plt.show()


# Random forest classifier
clf_rf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)
clf_rf.fit(X_train, y_train)
y_pred_rf = clf_rf.predict(X_test)

accuracy_rf = accuracy_score(y_test, y_pred_rf)
precision_rf = precision_score(y_test, y_pred_rf)
recall_rf = recall_score(y_test, y_pred_rf)
f1_rf = f1_score(y_test, y_pred_rf)
cm_rf = confusion_matrix(y_test, y_pred_rf)

# Neural network classifier
clf_nn = MLPClassifier(hidden_layer_sizes=(100, ), random_state=42)
clf_nn.fit(X_train, y_train) # X_train - dataset | y_train - datasets target
y_pred_nn = clf_nn.predict(X_test) # Predikter på dataset og target

accuracy_nn = accuracy_score(y_test, y_pred_nn)
precision_nn = precision_score(y_test, y_pred_nn)
recall_nn = recall_score(y_test, y_pred_nn)
f1_nn = f1_score(y_test, y_pred_nn)
cm_nn = confusion_matrix(y_test, y_pred_nn)


# Results
# M = dataset['relation' == 1]
# B = dataset['relation' == 0]
# models = ["Decision Tree", "Random Forest", "Neural Network"]
# accuracy = [accuracy_dt, accuracy_rf, accuracy_nn]
# plt.bar(models, accuracy)
# plt.title("Accuracy Scores for Breast Cancer Dataset")
# plt.xlabel("Model")
# plt.ylabel("Accuracy")
# plt.show()


print("Decision Tree Classification Results:")
print(f"Accuracy: {accuracy_dt*100:.3f}")
print(f"Precision: {precision_dt*100:.3f}")
print(f"Recall: {recall_dt*100:.3f}")
print(f"F1-score: {f1_dt*100:.3f}")
print(f"Confusion Matrix:\n{cm_dt}\n")

print("Random Forest Classification Results:")
print(f"Accuracy: {accuracy_rf*100:.3f}")
print(f"Precision: {precision_rf*100:.3f}")
print(f"Recall: {recall_rf*100:.3f}")
print(f"F1-score: {f1_rf*100:.3f}")
print(f"Confusion Matrix:\n{cm_rf}\n")

print("Neural Network Classification Results:")
print(f"Accuracy: {accuracy_nn*100:.3f}")
print(f"Precision: {precision_nn*100:.3f}")
print(f"Recall: {recall_nn*100:.3f}")
print(f"F1-score: {f1_nn*100:.3f}")
print(f"Confusion Matrix:\n{cm_nn}")
print(dataset.head().T)
# print(data.diagnosis.unique())
B, M = data['diagnosis'].value_counts()
print(M)
print(B)

# print("Cancer data set dimensions : {}".format(dataset.shape))
# Cancer data set dimensions : (569, 32)

# new data visualisation
# plt.rcParams['font.size']=10
# sns.pairplot(data, hue='Status', palette='Blues')

#Encoding categorical data values
# from sklearn.preprocessing import LabelEncoder
# labelencoder_Y = LabelEncoder()
# y = labelencoder_Y.fit_transform(Y)




# https://github.com/MuhammadBilalYar/ann-on-breast-cancer-dataset/blob/master/ml-breast-cancer-data-ann.py
# from ann_visualizer.visualize import ann_viz
# from keras.models import Sequential
# classifier = Sequential()
# classifier.fit(X_train, y_train, tatch_size=100, nb_epoch=150)
# y_pred = classifier.predict(X_test)
# y_pred = (y_pred > 0.5)
# # Making the Confusion Matrix
# from sklearn.metrics import confusion_matrix
# cm = confusion_matrix(y_test, y_pred)


# print("[Epoch:150] Our accuracy is {}%".format(((cm[0][0] + cm[1][1])/175)*100))

# sns.heatmap(cm,annot=True)
# plt.savefig('epoch150.png')
# ann_viz(classifier, title="Artificial Neural Network (ANN) implementation on Breast Cancer Wisconsin Data Set")