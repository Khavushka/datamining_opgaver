import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn import tree
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("Project 14\ASD.csv")
# Convert all values to float
# df = df.astype(float)
# df = df.apply(pd.to_numeric, errors='coerce')
# df.fillna(value=0, inplace=True)
# Define a dictionary of replacements for each column

replacements = {
    'gender': {'f': 1, 'm': 0},
    'ethnicity': {'White-European': 1, 'Latino': 2, 'Asian': 3, 'Hispanic': 4, 'Others': 5, 'Middle Eastern ':6, 'Black': 7, 'Turkish': 8, 'Pasifika': 9, 'South Asian': 10, 'others': 11},
    'jundice': {'yes': 1, 'no': 0},
    'austim': {'yes': 1, 'no': 0},
    'contry_of_res': {'Brazil': 1,'United States': 2, 'Spain': 3, 'India': 4, 'New Zealand': 5, 'Bahamas': 6, 'Bangladesh': 7, 'United Arab Emirates': 8, 'Burundi': 9, 'Jordan': 10, 'Ireland': 11, 'Afghanistan': 12, 'United Kingdom': 13, 'Canada': 14, 'Portugal': 15, 'Malaysia': 16, 'Belgium': 17, 'Poland': 18, 'Sri Lanka': 19, 'Pakistan': 20, 'Egypt': 21, 'Philippines': 22, 'Germany': 23, 'Australia': 24, 'Iran': 25, 'Ukraine': 26, 'Cyprus': 27, 'Mexico': 28, 'Nicaragua':29, 'Netherlands':30, 'Uruguay': 31, 'Nepal': 32, 'South Africa': 33, 'Chile': 34, 'Romania': 35, 'Rwanda': 36, 'Angola':37, 'Italy': 38, 'France': 39, 'Viet Nam': 40, 'Ethiopia': 41, 'AmericanSamoa': 42, 'Saudi Arabia': 43, 'Austria': 44, 'Armenia': 45, 'Indonesia': 46, 'Sweden': 47, 'Turkey': 48, 'Tonga': 49, 'Russia':50, 'Aruba':51, 'Costa Rica': 52, 'Czech Republic': 53, 'China': 54, 'Niger': 55, 'Bolivia': 56, 'Serbia': 57, 'Ecuador': 58, 'Finland': 59, 'Oman': 60, 'Sierra Leone': 61, 'Iceland':61},
    'used_app_before': {'yes': 1, 'no': 0},
    'age_desc': {'18 and more': 1, 'm': 0},
    'relation': {'Self': 1, 'Parent': 2, 'Relative': 3, 'Others':4, 'Health care professional':5},
    'Class/ASD': {'YES': 1, 'NO': 0},
}
# df = df.describe()
# df = df['jundice']
# Replace string values with 1's and 0's
df.replace(replacements, inplace=True)


# df = df.replace({'?': np.nan}).dropna()
# # Replace NaN values with the mean of the column
# # df['column_name'].fillna(df['column_name'].mean(), inplace=True)

# # Replace non-numeric values with NaN values
# df.replace("f", np.nan, inplace=True)
# df.replace("m", np.nan, inplace=True)
# df.replace("White-European", np.nan, inplace=True)
# df.replace("Latino", np.nan, inplace=True)


# Save the preprocessed dataset to a new file
df.to_csv("Project 14\ASD_new.csv", index=False)


# M = df['relation' == 1]
# B = df['relation' == 0]
# df.fillna(df.mean(), inplace=True)
# # Split the dataset into training and testing sets
X = df.drop(columns=["Class/ASD"])
y = df["Class/ASD"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(df.head())
print(df.describe())

# # Train and evaluate Decision Tree Classifier
dtc = tree.DecisionTreeClassifier()
dtc = dtc.fit(X_train, y_train)
fig = plt.figure(figsize=(25,20))
_ = tree.plot_tree(dtc, 
                    filled=True)
y_pred = dtc.predict(X_test)
dtc_acc = accuracy_score(y_test, y_pred)
print(f"Decision Tree Classifier Accuracy: {dtc_acc*100:.3f}")

# Train and evaluate Random Forest Classifier
rfc = RandomForestClassifier()
rfc.fit(X_train, y_train)
y_pred = rfc.predict(X_test)
rfc_acc = accuracy_score(y_test, y_pred)
print(f"Random Forest Classifier Accuracy: {rfc_acc*100:.3f}")

# Train and evaluate Neural Network Classifier
nnc = MLPClassifier(hidden_layer_sizes=(50, 50), max_iter=1000)
nnc.fit(X_train, y_train)
y_pred = nnc.predict(X_test)
nnc_acc = accuracy_score(y_test, y_pred)
print(f"Neural Network Classifier Accuracy: {nnc_acc*100:.3f}")


