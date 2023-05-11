import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


import pandas as pd

# Henter Breast Cancer dataset
url = "http://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data"

bcw_df = pd.read_csv(url, header=None)
# df.drop(0, axis=1, inplace=True) #dropper ID column
# X = df.iloc[:, 1:].values # separate target fra input 
# y = df.iloc[:, 0].values
# print(bcw.head())
bcw_X = bcw_df.iloc[:, 2:].values
bcw_y = bcw_df.iloc[:, 1].values

# Henter Bascketball dataset
bb_df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/00341/Huge%20Stock%20Market%20Dataset/basketball.csv')
bb_X = bb_df.iloc[:, :-1].values
bb_y = bb_df.iloc[:, -1].values

# Train 

# Test

# Logistisk regression



# Decision tree



# SVM - support vector machine