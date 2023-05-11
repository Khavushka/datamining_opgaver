import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


import pandas as pd

# Henter Breast Cancer dataset
bcw_df = pd.read_csv("http://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data", header=None)

# bcw_df = pd.read_csv(url, header=None)
# df.drop(0, axis=1, inplace=True) #dropper ID column
# X = df.iloc[:, 1:].values # separate target fra input 
# y = df.iloc[:, 0].values
bcw_X = bcw_df.iloc[:, 2:].values
bcw_y = bcw_df.iloc[:, 1].values
print(bcw_df.head())

# Henter Bascketball dataset
bb_df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/00341/Huge%20Stock%20Market%20Dataset/basketball.csv')
bb_X = bb_df.iloc[:, :-1].values
bb_y = bb_df.iloc[:, -1].values
print(bb_df.header())

# Spliter Breast Cancer Wisconsin dataset into training og testing sets
# bcw_X_train, bcw_X_test, bcw_y_train, bcw_y_test = train_test_split(bcw_X, bcw_y, test_size=0.2, random_state=42)

# Spliter Basketball dataset into training og testing sets
# bb_X_train, bb_X_test, bb_y_train, bb_y_test = train_test_split(bb_X, bb_y, test_size=0.2, random_state=42)

# Create and train a neural network model on the Breast Cancer Wisconsin dataset


# Create and train a support vector machine model on the Breast Cancer Wisconsin dataset


# Create and train a neural network model on the Basketball dataset


# Create and train a support vector machine model on the Basketball dataset



# Logistisk regression



# Decision tree



# SVM - support vector machine