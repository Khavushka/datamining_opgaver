import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix


# Load the Autism-Adult-Data dataset
data1 = pd.read_csv("http://archive.ics.uci.edu/ml/machine-learning-databases/00426/Autism-Adult-Data.arff", header=None)
# data1 = data1.drop([0, 19, 20, 21, 22], axis=1)  # Remove unnecessary columns
# data1 = data1.replace('?', np.nan).dropna()  # Remove missing values
# X1 = preprocessing.scale(data1.iloc[:, :-1].values)  # Normalize features
# y1 = data1.iloc[:, -1].values  # Target variable
print(data1)

# Load the Breast Cancer Wisconsin dataset
data2 = pd.read_csv("http://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data", header=None)
# data2 = data2.drop([0], axis=1)  # Remove unnecessary column
# X2 = preprocessing.scale(data2.iloc[:, 1:].values)  # Normalize features
# y2 = data2.iloc[:, 0].values  # Target variable
print(data2)





# df1 = df1.drop(['age_desc', 'result', 'age'], axis=1) # drop unnecessary columns
# df1 = df1.replace({'?': np.nan}).dropna() # drop rows with missing values
