'''
Neural networks and Support vector machines.
a. choose datasets of considerably difference size: what do you observe for the training time?
b. coose different setup for the classifiers:
    - what do you observe for the training time?
    - what do yoy observe for the apparent error and true error?
'''
from keras.datasets import mnist, cifar10
from keras.models import Sequential
from keras.layers import Dense
from sklearn.svm import SVC
import time

# Load MNIST dataset
(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = X_train.reshape(60000, 784).astype('float32') / 255
X_test = X_test.reshape(10000, 784).astype('float32') / 255

# Load CIFAR-10 dataset
(X_train_cifar, y_train_cifar), (X_test_cifar, y_test_cifar) = cifar10.load_data()
X_train_cifar = X_train_cifar.reshape(50000, 3072).astype('float32') / 255
X_test_cifar = X_test_cifar.reshape(10000, 3072).astype('float32') / 255

# Train neural network on MNIST
model = Sequential([
    Dense(128, activation='relu', input_shape=(784,)),
    Dense(10, activation='softmax')
])
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
start_time = time.time()
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))
print("Training time for MNIST neural network:", time.time() - start_time)

# Train SVM on MNIST
svm = SVC(kernel='linear')
start_time = time.time()
svm.fit(X_train, y_train)
print("Training time for MNIST SVM:", time.time() - start_time)

# Train neural network on CIFAR-10
model = Sequential([
    Dense(256, activation='relu', input_shape=(3072,)),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
start_time = time.time()
model.fit(X_train_cifar, y_train_cifar, epochs=10, batch_size=32, validation_data=(X_test_cifar, y_test_cifar))
print("Training time for CIFAR-10 neural network:", time.time() - start_time)

# Train SVM on CIFAR-10
svm = SVC(kernel='linear')
start_time = time.time()
svm.fit(X_train_cifar, y_train_cifar)
print("Training time for CIFAR-10 SVM:", time.time() - start_time)