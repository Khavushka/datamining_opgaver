# import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

# plot_tree(clf, 
#           feature_names=iris.get('feature_names'), 
#           class_names=iris.get('target_names'))

# plt.show()


import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder

# Define the dataset
dataset = pd.DataFrame({
    'Time since getting driving license': ['1-2 years', '2-7 years', '>7 years', '1-2 years', '>7 years', '1-2 years', '2-7 years', '2-7 years'],
    'Gender': ['male', 'male', 'female', 'female', 'male', 'male', 'female', 'male'],
    'Residential area': ['urban', 'rural', 'rural', 'rural', 'rural', 'rural', 'urban', 'urban'],
    'Risk class': ['low', 'high', 'low', 'high', 'high', 'high', 'low', 'low']
})
print(dataset)

# Preprocess the dataset
label_encoder = LabelEncoder()
for column in dataset.columns:
    dataset[column] = label_encoder.fit_transform(dataset[column])
    # print(column)

# Split dataset
X = dataset.drop('Risk class', axis=1)
y = dataset['Risk class']

# Define and train the decision tree classifier
decision_tree = DecisionTreeClassifier(criterion='entropy')
decision_tree.fit(X, y)

# Predict the risk class 
new_instance = [[1, 0, 1]] # 1-2 years, male, urban

prediction = decision_tree.predict(new_instance)
print(prediction)