import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LeakyReLU, PReLU, ELU
from keras.layers import Dropout
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

# load dataset
df = pd.read_csv('Project 14/data.csv')
value_counts = df['diagnosis'].value_counts()
print(value_counts)
# independent variables
x = df.drop('diagnosis', axis=1)
# dependent variables
y = df.diagnosis

# creating the object
lb = LabelEncoder()
y = lb.fit_transform(y)

from sklearn.model_selection import train_test_split
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.3, random_state=40)

# scaling the data
# creating object
sc = StandardScaler()
xtrain = sc.fit_transform(xtrain)
xtest = sc.transform(xtest)

'''
oprettelse af lag:
- input lag
- hidden lag
- output lag
Først laver man modellen
'''
classifier = Sequential()
# first hidden layer
classifier.add(Dense(units=9, kernel_initializer='he_uniform', activation='relu', input_dim=32))

# second hidden layer
classifier.add(Dense(units=9, kernel_initializer='he_uniform', activation='relu'))

# sidste lag eller outputlag
classifier.add(Dense(units=1, kernel_initializer='glorot_uniform', activation='sigmoid'))

classifier.summary()

# kompilere ANN
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model = classifier.fit(xtrain, ytrain, batch_size=100, epochs=100)

# nu tester for testdata
y_pred = classifier.predict(xtest)

# converting values
y_pred = (y_pred > 0.5)
print(y_pred)

# score og confusion matrix
cm = confusion_matrix(ytest, y_pred)
score = accuracy_score(ytest, y_pred)
print(cm)
print('score is:', score)

# vusualisation confusion matrix
plt.figure(figsize=[14, 7])
sb.heatmap(cm, annot=True)
plt.show()

# visualisation data history
print(model.history.keys())
# summarize history for accuracy
plt.plot(model.history['accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(model.history['loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# saving model
classifier.save('neural_network.h5')
