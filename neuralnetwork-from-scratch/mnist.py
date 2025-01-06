import numpy as np

from model import Network
from dense import Dense
from activation import ActivationLayer
from activations import tanh, tanh_prime
from losses import mse, mse_prime

from keras._tf_keras.keras.datasets import mnist
from keras._tf_keras.keras.utils import to_categorical

# load MNIST from server
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, y_train = x_train[:1000], y_train[:1000]
x_test, y_test = x_test[:100], y_test[:100]

# training data : 60000 samples
# reshape and normalize input data, remember that input data is COLUMN VECTOR
x_train = x_train.reshape(x_train.shape[0], 28 * 28, 1) #Column vectors input
x_train = x_train.astype('float32')
x_train /= 255
# encode output which is a number in range [0,9] into a vector of size 10
# e.g. number 3 will become [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
y_train = to_categorical(y_train)
y_train = y_train.reshape(y_train.shape[0], 10, 1) #Column vectors onehot encoding target

# same for test data : 10000 samples
x_test = x_test.reshape(x_test.shape[0], 28 * 28, 1)
x_test = x_test.astype('float32')
x_test /= 255
y_test = to_categorical(y_test)
y_test = y_test.reshape(y_test.shape[0], 10, 1)

net = Network()
net.add(Dense(28 * 28, 100))  # input_shape=(28 * 28, 1)    ;   output_shape=(100, 1)
net.add(ActivationLayer(tanh, tanh_prime))
net.add(Dense(100, 50))  # input_shape=(100, 1)      ;   output_shape=(50, 1)
net.add(ActivationLayer(tanh, tanh_prime))
net.add(Dense(50, 10))  # input_shape=(50, 1)       ;   output_shape=(10, 1)
net.add(ActivationLayer(tanh, tanh_prime))

# train on 1000 samples
net.use(mse, mse_prime)
net.fit(x_train[0:1000], y_train[0:1000], epochs=10, learning_rate=0.1)

# test on 3 samples
out = net.predict(x_test[0:3])
print("\n")
print("predicted values : ")
print(out, end="\n")
print("true values : ")
print(y_test[0:3])
