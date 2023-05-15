# import numpy as np
# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# import tensorflow as tf
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense
# from sklearn.metrics import confusion_matrix

# # Load the breast cancer dataset (example: using the Wisconsin Breast Cancer dataset)
# url = "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data"
# column_names = ['id', 'diagnosis', 'mean_radius', 'mean_texture', 'mean_perimeter', 'mean_area', 'mean_smoothness',
#                 'mean_compactness', 'mean_concavity', 'mean_concave_points', 'mean_symmetry', 'mean_fractal_dimension',
#                 'se_radius', 'se_texture', 'se_perimeter', 'se_area', 'se_smoothness', 'se_compactness', 'se_concavity',
#                 'se_concave_points', 'se_symmetry', 'se_fractal_dimension', 'worst_radius', 'worst_texture',
#                 'worst_perimeter', 'worst_area', 'worst_smoothness', 'worst_compactness', 'worst_concavity',
#                 'worst_concave_points', 'worst_symmetry', 'worst_fractal_dimension']
# data = pd.read_csv(url, names=column_names)

# # Preprocessing the data
# X = data.iloc[:, 2:].values  # Feature variables
# y = data['diagnosis'].map({'M': 1, 'B': 0})  # Target variable
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# scaler = StandardScaler()
# X_train = scaler.fit_transform(X_train)
# X_test = scaler.transform(X_test)

# # Build the neural network model
# model = Sequential()
# model.add(Dense(16, activation='relu', input_shape=(30,)))
# model.add(Dense(8, activation='relu'))
# model.add(Dense(1, activation='sigmoid'))

# model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# # Train the model
# model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=1)

# # Evaluate the model
# _, accuracy = model.evaluate(X_test, y_test, verbose=0)
# print('Accuracy:', accuracy)

# import matplotlib.pyplot as plt
# from sklearn.metrics import ConfusionMatrixDisplay, roc_curve, auc

# # Generate predictions on the test set
# y_pred = model.predict(X_test)
# y_pred = (y_pred > 0.5)

# # Confusion Matrix
# cm = confusion_matrix(y_test, y_pred)
# labels = ['Benign', 'Malignant']
# display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
# display.plot(cmap='Blues')
# plt.title('Confusion Matrix')
# plt.show()

# # ROC Curve
# fpr, tpr, thresholds = roc_curve(y_test, y_pred)
# roc_auc = auc(fpr, tpr)
# plt.plot(fpr, tpr, color='blue', label='AUC = %0.2f' % roc_auc)
# plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
# plt.xlim([0.0, 1.0])
# plt.ylim([0.0, 1.05])
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('Receiver Operating Characteristic (ROC) Curve')
# plt.legend(loc="lower right")
# plt.show()

# from sklearn.metrics import precision_recall_curve

# precision, recall, thresholds = precision_recall_curve(y_test, y_pred)
# plt.plot(recall, precision, color='blue')
# plt.xlabel('Recall')
# plt.ylabel('Precision')
# plt.title('Precision-Recall Curve')
# plt.show()

# from sklearn.metrics import classification_report

# report = classification_report(y_test, y_pred, target_names=labels)
# print(report)


# import seaborn as sns

# corr_matrix = data.iloc[:, 1:].corr()
# plt.figure(figsize=(12, 8))
# sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
# plt.title('Correlation Matrix Heatmap')
# plt.show()


# importances = model.coef_[0]
# feature_names = data.columns[2:]
# sorted_indices = np.argsort(importances)[::-1]
# sorted_importances = importances[sorted_indices]
# sorted_feature_names = feature_names[sorted_indices]

# plt.figure(figsize=(12, 6))
# plt.bar(range(len(importances)), sorted_importances)
# plt.xticks(range(len(importances)), sorted_feature_names, rotation='vertical')
# plt.xlabel('Features')
# plt.ylabel('Importance')
# plt.title('Feature Importance')
# plt.show()


# --------------------------------------------------
from sklearn.datasets import load_breast_cancer
# import numpy as np
# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# import tensorflow as tf
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense

# # Load the Breast Cancer dataset
# data = load_breast_cancer()

# # Extract features and target variable
# X = data.data  # Input features
# y = data.target  # Target variable (diagnostic outcome)

# # Split the data into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Scale the feature values
# scaler = StandardScaler()
# X_train = scaler.fit_transform(X_train)
# X_test = scaler.transform(X_test)

# # Build the ANN model
# model = Sequential()

# # Input layer
# model.add(Dense(8, activation='relu', input_shape=(X.shape[1],)))

# # Hidden layers
# model.add(Dense(16, activation='relu'))
# model.add(Dense(8, activation='relu'))

# # Output layer
# model.add(Dense(1, activation='sigmoid'))

# # Compile the model
# model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# # Train the model
# model.fit(X_train, y_train, epochs=50, batch_size=32)

# # Evaluate the model
# _, accuracy = model.evaluate(X_test, y_test)
# print('Accuracy:', accuracy)

# --------------------------------------------------------------------


# Importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Importing data
data = load_breast_cancer()
# del data['Unnamed: 32']

#Feature and Label selection
X = data.iloc[:, 2:].values
y = data.iloc[:, 1].values

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder
labelencoder_X_1 = LabelEncoder()
y = labelencoder_X_1.fit_transform(y)

# Splitting the dataset into the Training set and Test set (30 - 70)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)

#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout

# Initialising the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense(output_dim=16, init='uniform', activation='relu', input_dim=30))
# Adding dropout to prevent overfitting
classifier.add(Dropout(p=0.1))


# Adding the second hidden layer
classifier.add(Dense(output_dim=16, init='uniform', activation='relu'))
# Adding dropout to prevent overfitting
classifier.add(Dropout(p=0.1))


# Adding the output layer (output_dim is 1 as we want only 1 output from the final layer.)
classifier.add(Dense(output_dim=1, init='uniform', activation='sigmoid'))

# Compiling the ANN
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Fitting the ANN to the Training set
classifier.fit(X_train, y_train, batch_size=100, nb_epoch=50)

# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)


print("[Epoch:50] Our accuracy is {}%".format(((cm[0][0] + cm[1][1])/175)*100))

sns.heatmap(cm,annot=True)
plt.savefig('epoch50.png')

# Fitting the ANN to the Training set
classifier.fit(X_train, y_train, batch_size=100, nb_epoch=100)


# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)


print("[Epoch:100] Our accuracy is {}%".format(((cm[0][0] + cm[1][1])/175)*100))

sns.heatmap(cm,annot=True)
plt.savefig('epoch100.png')

# Fitting the ANN to the Training set
classifier.fit(X_train, y_train, batch_size=100, nb_epoch=150)


# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)


print("[Epoch:150] Our accuracy is {}%".format(((cm[0][0] + cm[1][1])/175)*100))

sns.heatmap(cm,annot=True)
plt.savefig('epoch150.png')


from ann_visualizer.visualize import ann_viz;

ann_viz(classifier, title="Artificial Neural Network (ANN) implementation on Breast Cancer Wisconsin Data Set")