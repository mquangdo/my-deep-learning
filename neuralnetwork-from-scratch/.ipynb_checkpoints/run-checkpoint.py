import numpy as np

from model import Network
from dense import Dense
from activation import ActivationLayer
from activations import tanh, tanh_prime
from losses import mse, mse_prime

# training data
x_train = np.array([[[0], [0]], [[0], [1]], [[1], [0]], [[1], [1]]])
y_train = np.array([[[0]], [[1]], [[1]], [[0]]])

print(x_train[0].shape)
print(y_train[0].shape)

# network
net = Network()
net.add(Dense(2, 3))
net.add(ActivationLayer(tanh, tanh_prime))
net.add(Dense(3, 1))
net.add(ActivationLayer(tanh, tanh_prime))

# train
net.use(mse, mse_prime)
net.fit(x_train, y_train, epochs=10000, learning_rate=0.1)

# test
out = net.predict(x_train)
print(out)